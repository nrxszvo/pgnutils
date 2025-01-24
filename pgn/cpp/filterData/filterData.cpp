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
namespace fs = std::filesystem;

ABSL_FLAG(std::string, npydir, "", "directory containing block folders of npy raw data files");
ABSL_FLAG(int, minMoves, 11, "minimum number of game moves to be included in filtered dataset");
ABSL_FLAG(int, minTime, 30, "minimum time remaining to be included in filtered dataset");
ABSL_FLAG(std::string, outdir, "", "output directory for writing memmap files");
ABSL_FLAG(float, trainp, 0.9, "percentage of dataset for training");
ABSL_FLAG(float, testp, 0.08, "percentage of dataset for testing");
ABSL_FLAG(std::vector<std::string>, eloEdges, std::vector<std::string>({"1000","1200","1400","1600","1800","2000","2200","2400","2600"}), "ELO bin edges for ensuring even distribution of ELOs");
ABSL_FLAG(int, maxGamesPerElo, -1, "maximum number of games per ELO group (-1 to disable)");
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
	std::string fp;
	std::ofstream idxData;
	std::string name;
	Split(std::string name, std::string outdir): name(name), nSamp(0), nGames(0), fp(outdir + "/" + name + ".npy") {
	   	idxData = std::ofstream(fp, std::ios::binary);
	}
};

class SplitManager {
	std::vector<std::vector<Split>> splits;
	std::vector<std::string> names;
	int16_t maxElo;
	int minMoves;
	RandDS rds;
	std::string outdir;
public:
	struct Args {
		std::string outdir;
		int nThreads;
		std::vector<std::string> names;
		float trainp;
		float testp;
		int minMoves;
	};
	SplitManager(Args& args)
		: names(args.names), maxElo(0), minMoves(args.minMoves), rds(args.trainp, args.testp), outdir(args.outdir) 
	{
		splits = std::vector<std::vector<Split>>(args.nThreads);
		for (int threadId=0; threadId < args.nThreads; threadId++) {
			for (auto name: names) {
				std::string tName = "thread-" + std::to_string(threadId) + "-" + name;
				splits[threadId].push_back(Split(tName, outdir));
			}
		}
	}
	void insertCoords(int threadId, int64_t gIdx, int64_t gStart, int64_t gLength, int64_t blockId, int16_t welo, int16_t belo){
		maxElo = std::max(maxElo, std::max(welo, belo));
		int dsIdx = rds.get();
		
		splits[threadId][dsIdx].idxData.write((char*)&gIdx, sizeof(int64_t));
		splits[threadId][dsIdx].idxData.write((char*)&gStart, sizeof(int64_t));
		splits[threadId][dsIdx].idxData.write((char*)&gLength, sizeof(int64_t));
		splits[threadId][dsIdx].idxData.write((char*)&blockId, sizeof(int64_t));
		splits[threadId][dsIdx].nGames++;
	};
	int64_t getNGames() {
		int64_t nGames = 0;
		for (auto& ttv: splits) {
			for (auto& split: ttv) {
				nGames += split.nGames;
			}
		}
		return nGames;
	}
	int16_t getMaxElo() {
		return maxElo;
	}
	void finalizeData(std::vector<std::string>& blockDirs) {
		json md;
		md["ngames"] = getNGames();
		md["min_moves"] = minMoves;
		md["block_dirs"] = blockDirs;	
		std::cout << std::endl << "Consolidating output data..." << std::endl;
		for (int i=0; i<splits[0].size(); i++) {
			int64_t nGames = 0;
			std::ofstream consOf = std::ofstream(outdir + "/" + names[i] + ".npy", std::ios::binary);
			for (int threadId=0; threadId < splits.size(); threadId++) {
				Split& split = splits[threadId][i];
				nGames += split.nGames;
				split.idxData.close();
				std::ifstream thrIf = std::ifstream(split.fp, std::ios::binary);
				consOf << thrIf.rdbuf();
				thrIf.close();
				fs::remove(split.fp);
			}
			consOf.close();
			md[names[i] + "_shape"] = {nGames,4};
			md[names[i] + "_n"] = nGames;
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
	std::vector<int64_t> blockGames;
	std::vector<std::shared_ptr<MMCRawDataReader>> readers;
public:
	struct Args {
		int nBlocks;
		int nThreadsPerBlock;
		int minMoves;
		int minTime;
		int maxGames;
		int maxGamesLeniency;
		std::vector<int> eloEdges;
		SplitManager& splitMgr;
		Args(SplitManager& splitMgr): splitMgr(splitMgr) {};
	};
	BlockProcessor(Args& args)
		: minMoves(args.minMoves), minTime(args.minTime), maxGames(args.maxGames), leniency(args.maxGamesLeniency), eloEdges(args.eloEdges), splitMgr(args.splitMgr) 
	{
		eloHist = std::vector(eloEdges.size(), std::vector(eloEdges.size(), 0));
		blockGames = std::vector<int64_t>(args.nBlocks*args.nThreadsPerBlock, 0);
	}

	int64_t completedGames() {
		int64_t total = 0;
		for (auto count: blockGames) {
			total += count;
		}
		return total;
	}

	std::vector<std::vector<int>>& getEloHist() { return eloHist; }

	void addReader(std::shared_ptr<MMCRawDataReader> mrd) {
		readers.push_back(mrd);
	}

	int64_t totalGamesEstimate() {
		int64_t totalGames = 0;
		for (auto mrd: readers) {
			totalGames += mrd->getTotalGames();
		}
		return totalGames;
	}

	void processBlock(int blockId, int threadId) {
		std::vector<int16_t> clk;
		auto localHist = std::vector(eloEdges.size(), std::vector(eloEdges.size(), 0));
		auto mrd = readers[threadId];
		while (true) {
			auto [bytesRead, gIdx, gameStart, whiteElo, blackElo] = mrd->nextGame(clk);
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
			if (maxGames > 0 && eloHist[wbin][bbin] >= maxGames) continue;

			int idx = clk.size()-1;	
			while (idx >= minMoves && clk[idx] < minTime && clk[idx-1] < minTime) idx--;

			if (idx >= minMoves) {
				localHist[wbin][bbin]++;
				if (maxGames > 0 && blockGames[threadId] % leniency == 0) {
					std::lock_guard<std::mutex> lock(histoMtx);
					for (int i=0; i<eloHist.size(); i++) {
						for (int j=0; j<eloHist.size(); j++) {
							eloHist[i][j] += localHist[i][j];
							localHist[i][j] = 0;
						}
					}
				}
				splitMgr.insertCoords(threadId, gIdx, gameStart, idx+1, blockId, whiteElo, blackElo);
			}
		}
	}
};

void filterData(std::vector<std::string>& npydir, int nThreadsPerBlock, int minMoves, int minTime, std::string& outdir, float trainp, float testp, std::vector<int>& eloEdges, int maxGames, int maxGamesLeniency) {
	std::vector<int64_t> gamesPerThread;
	for (int blkId=0; blkId < npydir.size(); blkId++) {
		MMCRawDataReader mrd = MMCRawDataReader(npydir[blkId]);
		int64_t blockGames = mrd.getTotalGames();
		double gpt = (double)blockGames/nThreadsPerBlock;
		gamesPerThread.push_back((int64_t)(std::ceil(gpt)));
	}

	SplitManager::Args smArgs;
	smArgs.outdir = outdir;
	smArgs.nThreads = npydir.size()*nThreadsPerBlock;
	smArgs.names = {"train", "val", "test"};
	smArgs.trainp = trainp;
	smArgs.testp = testp;
	smArgs.minMoves = minMoves;

	SplitManager splitMgr = SplitManager(smArgs);

	BlockProcessor::Args bpArgs(splitMgr);
	bpArgs.nBlocks = npydir.size();
	bpArgs.nThreadsPerBlock = nThreadsPerBlock;
	bpArgs.minMoves = minMoves;
	bpArgs.minTime = minTime;
	bpArgs.maxGames = maxGames;
	bpArgs.maxGamesLeniency = maxGamesLeniency;
	bpArgs.eloEdges = eloEdges;

	BlockProcessor blkProc(bpArgs);

	std::vector<std::shared_ptr<std::thread> > threads;
	auto processBlock = [](BlockProcessor& blkProc, int blkId, int threadId) {
		return blkProc.processBlock(blkId, threadId);
	};

	for (int blockId = 0; blockId < npydir.size(); blockId++) {
		int64_t nGames = gamesPerThread[blockId];
		for (int i=0; i<nThreadsPerBlock; i++) {
			if (i == nThreadsPerBlock-1) nGames = -1;
			int64_t startGame = i*gamesPerThread[blockId];
			auto mrd = std::make_shared<MMCRawDataReader>(npydir[blockId], startGame, nGames);
			blkProc.addReader(mrd);

			threads.push_back(std::make_shared<std::thread>(
				processBlock, 
				std::ref(blkProc), 
				blockId, 
				blockId*nThreadsPerBlock+i
			));
		}
	}

	auto start = hrc::now();
	while (true) {
		std::this_thread::sleep_for(500ms);
		int64_t completedGames = blkProc.completedGames();
		int64_t totalGamesEst = blkProc.totalGamesEstimate();
		int complete = (int)(100*(double)completedGames / totalGamesEst);
		auto [eta, now] = getEta(totalGamesEst, completedGames, start); 
		std::cout << completedGames << " out of " << totalGamesEst << " (" << complete << "% done, eta: " << eta << ")\r" << std::flush;
		if (complete == 100) break;
	}
	for (auto thread: threads) {
		thread->join();
	}
	splitMgr.finalizeData(npydir);
	printReport(splitMgr.getNGames(), blkProc.completedGames(), eloEdges, blkProc.getEloHist(), splitMgr.getMaxElo()); 

	auto stop = hrc::now();	
	auto ellapsed = getEllapsedStr(start, stop);
	std::cout << "Total time: " << ellapsed << std::endl;
}

std::vector<std::string> getBlockDirs(const std::string& npydir)
{
	const re2::RE2 BLOCK_PAT = ".*block-([0-9]+).*";
	int64_t blockId;
	int nBlocks = 0;
	for(auto& p : fs::recursive_directory_iterator(npydir)) {
		if (p.is_directory()) {
			std::string dn = p.path().string();
			if (re2::RE2::PartialMatch(dn, BLOCK_PAT, &blockId)) {
				nBlocks++;
			}
		}
	}
	std::vector<std::string> blockDns(nBlocks);
	for(auto& p : fs::recursive_directory_iterator(npydir)) {
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
