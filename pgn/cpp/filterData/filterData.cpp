#include "MMCRawDataReader.h"
#include "utils/utils.h"
#include "npy.hpp"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "json.hpp"
#include <absl/flags/internal/flag.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <filesystem>
#include <tuple>
#include <re2/re2.h>
#include <mutex>
#include <thread>

using json = nlohmann::json;
using namespace std::chrono_literals;

ABSL_FLAG(std::string, npydir, "", "directory containing block folders of npy raw data files");
ABSL_FLAG(int, minMoves, 11, "minimum number of game moves to be included in filtered dataset");
ABSL_FLAG(int, minTime, 30, "minimum time remaining to be included in filtered dataset");
ABSL_FLAG(std::string, outdir, "", "output directory for writing memmap files");
ABSL_FLAG(float, trainp, 0.9, "percentage of dataset for training");
ABSL_FLAG(float, testp, 0.08, "percentage of dataset for testing");
ABSL_FLAG(std::vector<std::string>, eloEdges, std::vector<std::string>({"1000","1200","1400","1600","1800","2000","2200","2400","2600"}), "ELO bin edges for ensuring even distribution of ELOs");
ABSL_FLAG(int, maxGamesPerElo, 5000000, "maximum number of games per ELO group");
ABSL_FLAG(int, nThreadsPerBlock, 1, "number of threads per block");
ABSL_FLAG(int, maxGamesLeniency, 100, "allow maxGamesPerElo to be exceeded by approx. this number in order to greatly speed up parallel processing");

class RandDS {
	std::random_device rd;
	std::mt19937 e2;
	std::uniform_real_distribution<> dist;
	float trainp;
	float testp;
public:
	RandDS(float trainp, float testp): e2(rd()), dist(0,1), trainp(trainp), testp(testp) {};	

	int get() {
		float r = dist(e2);	
		if (r < trainp) return 0;
		else if (r < (trainp + testp)) return 1;
		else return 2;
	}
};

struct Split {
	int64_t nSamp;
	int64_t nGames;
	std::ofstream idxData;
	std::string name;
	Split(std::string name): name(name), nSamp(0), nGames(0) {};
};

class SplitManager {
	std::vector<Split> splits;
	int16_t maxElo;
	int64_t nGames; 
	int minMoves;
	RandDS rds;
	std::ofstream gsfile;
	std::ofstream elofile;
	std::string outdir;
public:
	SplitManager(std::string& outdir, std::vector<std::string>& names, float trainp, float testp, int minMoves)
		: gsfile(outdir + "/gs.npy", std::ios::binary), elofile(outdir + "/elo.npy", std::ios::binary), maxElo(0), nGames(0), minMoves(minMoves), rds(trainp, testp), outdir(outdir) 
	{
		for (auto name: names) {
			splits.push_back(Split(name));
		}
	}
	auto insertCoords(int64_t gStart, int64_t gLength, int16_t welo, int16_t belo, int64_t blockId){
		maxElo = std::max(maxElo, std::max(welo, belo));
		int dsIdx = rds.get();
		
		splits[dsIdx].idxData.write((char*)&gStart, sizeof(int64_t));
		splits[dsIdx].idxData.write((char*)&gLength, sizeof(int64_t));
		splits[dsIdx].idxData.write((char*)&nGames, sizeof(int64_t));
		splits[dsIdx].idxData.write((char*)&blockId, sizeof(int64_t));
		splits[dsIdx].nGames++;

		gsfile.write((char*)&gStart, sizeof(gStart));
		elofile.write((char*)&welo, sizeof(welo));
		elofile.write((char*)&belo, sizeof(belo));

		nGames++;
	};
	int64_t getNGames() {
		return nGames;
	}
	int16_t getMaxElo() {
		return maxElo;
	}
	void finalizeData(std::vector<std::string>& blockDirs) {
		gsfile.close();
		elofile.close();

		json md;
		md["ngames"] = nGames;
		md["min_moves"] = minMoves;
		md["block_dirs"] = blockDirs;	
		for (int i=0; i<splits.size(); i++) {
			Split& split = splits[i];
			split.idxData.close();
			md[split.name + "_shape"] = {split.nGames,4};
			md[split.name + "_n"] = split.nGames;
		}
		std::ofstream mdfile(outdir + "/fmd.json");
		mdfile << md << std::endl;
	}
};

void printReport(int64_t nInc, int64_t nTotal, std::vector<int>& eloEdges, std::vector<std::vector<int>> &eloHist, int16_t maxElo) {
	std::cout << "Included " << nInc << " out of " << nTotal << " games" << std::endl;
	eloEdges[eloEdges.size()-1] = maxElo;
	for (int i=eloEdges.size()-1; i>=0; i--) {
		std::cout << std::setfill(' ') << std::setw(11) << eloEdges[i];
		for (int j=0; j<eloEdges.size(); j++) {
			std::cout << std::setfill(' ') << std::setw(11) << eloHist[i][j];
		}
		std::cout << std::endl;
	}
	std::cout << std::setfill(' ') << std::setw(11) << ' ';
	for (auto e: eloEdges) {
		std::cout << std::setfill(' ') << std::setw(11) << e;
	}
	std::cout << std::endl;
}


auto getEloBin(int elo, std::vector<int>& eloEdges) {
	for (int i=0; i<eloEdges.size(); i++) {
		if (eloEdges[i] > elo) {
			return i;
		}
	}
	return -1;
};


class BlockProcessor {
	int minMoves;
	int minTime;
	int maxGames;
	int leniency;
	std::vector<int>& eloEdges;
	std::vector<std::vector<int>> eloHist;
	SplitManager& splitMgr;
	std::mutex histoMtx;
	std::mutex insertMtx;
	std::vector<int64_t> blockGames;
public:
	BlockProcessor(int nBlocks, int nThreadsPerBlock, int minMoves, int minTime, int maxGames, int maxGamesLeniency, std::vector<int>& eloEdges, SplitManager& splitMgr)
		: minMoves(minMoves), minTime(minTime), maxGames(maxGames), leniency(maxGamesLeniency), eloEdges(eloEdges), splitMgr(splitMgr) 
	{
		eloHist = std::vector(eloEdges.size(), std::vector(eloEdges.size(), 0));
		blockGames = std::vector(nBlocks*nThreadsPerBlock, (int64_t)0);
	}

	int64_t totalGames() {
		int64_t total = 0;
		for (auto count: blockGames) {
			total += count;
		}
		return total;
	}

	std::vector<std::vector<int>>& getEloHist() { return eloHist; }

	void processBlock(std::string& blkFn, int blockId, int threadId, int64_t startGame, int64_t nGames) {
		auto mrd = MMCRawDataReader(blkFn, startGame, nGames);
		std::vector<int16_t> clk;
		int64_t gamesSoFar = 0;
		std::vector<std::vector<int>> localHist = std::vector(eloEdges.size(), std::vector(eloEdges.size(), 0));

		while (true) {
			auto [bytesRead, gameStart, whiteElo, blackElo] = mrd.nextGame(clk);
			if (bytesRead == 0) { 
				std::lock_guard<std::mutex> lock(histoMtx);
				for (int i=0; i<eloHist.size(); i++) {
					for (int j=0; j<eloHist.size(); j++) {
						eloHist[i][j] += localHist[i][j];
					}
				}
				break;
			}

			blockGames[threadId]++;

			int wbin = getEloBin(whiteElo, eloEdges);
			int bbin = getEloBin(blackElo, eloEdges);
			if (eloHist[wbin][bbin] >= maxGames) continue;

			localHist[wbin][bbin]++;
			if (blockGames[threadId] % leniency == 0) {
				std::lock_guard<std::mutex> lock(histoMtx);
				for (int i=0; i<eloHist.size(); i++) {
					for (int j=0; j<eloHist.size(); j++) {
						eloHist[i][j] += localHist[i][j];
						localHist[i][j] = 0;
					}
				}
			}
			
			int idx = clk.size()-1;	
			while (idx >= minMoves && clk[idx] < minTime && clk[idx-1] < minTime) idx--;
			if (idx >= minMoves) {
				std::lock_guard<std::mutex> lock(insertMtx);
				splitMgr.insertCoords(gameStart, idx+1, whiteElo, blackElo, blockId);
			}
		}
	}
};

void filterData(std::vector<std::string>& npydir, int nThreadsPerBlock, int minMoves, int minTime, std::string& outdir, float trainp, float testp, std::vector<int>& eloEdges, int maxGames, int maxGamesLeniency) {
	size_t totalGames = 0;
	std::vector<int64_t> gamesPerThread;
	std::cout.precision(2);
	for (int blkId=0; blkId < npydir.size(); blkId++) {
		MMCRawDataReader mrd = MMCRawDataReader(npydir[blkId]);
		std::cout << "Block " << blkId << ": " << (float)mrd.getTotalGames() << " games" << std::endl;
		gamesPerThread.push_back(mrd.getTotalGames()/nThreadsPerBlock);
		totalGames += mrd.getTotalGames();
	}
	std::cout << "Total games: " << (float)totalGames << std::endl;

	std::vector<std::string> names = {"train", "val", "test"};
	SplitManager splitMgr = SplitManager(outdir, names, trainp, testp, minMoves);
	BlockProcessor blkProc(npydir.size(), nThreadsPerBlock, minMoves, minTime, maxGames, maxGamesLeniency, eloEdges, splitMgr);

	std::vector<std::shared_ptr<std::thread> > threads;
	auto processBlock = [&](BlockProcessor& blkProc, std::string dn, int blkId, int threadId, int64_t startGame, int64_t nGames) {
		return blkProc.processBlock(dn, blkId, threadId, startGame, nGames);
	};

	for (int blockId = 0; blockId < npydir.size(); blockId++) {
		for (int i=0; i<nThreadsPerBlock; i++) {
			int64_t nGames = gamesPerThread[blockId];
			if (i == nThreadsPerBlock-1) nGames = -1;
			threads.push_back(std::make_shared<std::thread>(
				processBlock, 
				std::ref(blkProc), 
				npydir[blockId], 
				blockId, 
				blockId*nThreadsPerBlock+i,
				i*gamesPerThread[blockId], nGames 
			));
		}
	}

	auto start = hrc::now();
	auto last = start;
	auto remaining = totalGames;
	int64_t lastCompleted = 0;
	while (true) {
		std::this_thread::sleep_for(5000ms);
		int64_t completedGames = blkProc.totalGames();
		int complete = (int)(100*(float)completedGames / totalGames);
		auto [eta, now] = getEta(remaining, completedGames-lastCompleted, last); 
		remaining -= (completedGames-lastCompleted);
		lastCompleted = completedGames;
		last = now;
		std::cout << complete << "% done (eta: " << eta << ")\r" << std::flush;
		if (complete >= 99) break;
	}
	for (auto thread: threads) {
		thread->join();
	}
	splitMgr.finalizeData(npydir);
	printReport(splitMgr.getNGames(), blkProc.totalGames(), eloEdges, blkProc.getEloHist(), splitMgr.getMaxElo()); 

	auto stop = hrc::now();	
	auto ellapsed = getEllapsedStr(start, stop);
	std::cout << "Total time: " << ellapsed << std::endl;
}

std::vector<std::string> getBlockDirs(const std::string& npydir)
{
	const re2::RE2 BLOCK_PAT = ".*block-([0-9]+).*";
	int64_t blockId;
	int nBlocks = 0;
	for(auto& p : std::filesystem::recursive_directory_iterator(npydir)) {
		if (p.is_directory()) {
			std::string dn = p.path().string();
			if (re2::RE2::PartialMatch(dn, BLOCK_PAT, &blockId)) {
				nBlocks++;
			}
		}
	}
	std::vector<std::string> blockDns(nBlocks);
	for(auto& p : std::filesystem::recursive_directory_iterator(npydir)) {
		if (p.is_directory()) {
			std::string dn = p.path().string();
			if (re2::RE2::PartialMatch(dn, BLOCK_PAT, &blockId)) {
				blockDns[blockId] = dn;
			}
		}
	}
	return blockDns;
}



int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("filter raw MimicChess dataset based on minimum number-of-moves and time-remaining constraints; randomly assign each game to train, val, or test sets");
	absl::ParseCommandLine(argc, argv);
	std::string npydir = absl::GetFlag(FLAGS_npydir);
	int minMoves = absl::GetFlag(FLAGS_minMoves);
	int minTime = absl::GetFlag(FLAGS_minTime);
	std::string outdir = absl::GetFlag(FLAGS_outdir);
	float trainp = absl::GetFlag(FLAGS_trainp);
	float testp = absl::GetFlag(FLAGS_testp);
	std::vector<std::string> eloEdgeStr = absl::GetFlag(FLAGS_eloEdges);
	std::vector<int> eloEdges;
	for (auto e: eloEdgeStr) {
		eloEdges.push_back(std::stoi(e));
	}
	eloEdges.push_back(INT_MAX);
	int maxGames = absl::GetFlag(FLAGS_maxGamesPerElo);
	int maxGamesLeniency = absl::GetFlag(FLAGS_maxGamesLeniency);
	std::vector<std::string>blockDirs = getBlockDirs(npydir);
	int nThreadsPerBLock = absl::GetFlag(FLAGS_nThreadsPerBlock);
	filterData(blockDirs, nThreadsPerBLock, minMoves, minTime, outdir, trainp, testp, eloEdges, maxGames, maxGamesLeniency);
	return 0;
}
