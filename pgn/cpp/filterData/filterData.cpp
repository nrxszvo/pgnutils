#include "MMCRawDataReader.h"
#include "npy.hpp"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <filesystem>

using json = nlohmann::json;

ABSL_FLAG(std::string, npydir, "", "directory containing npy raw data files");
ABSL_FLAG(int, minMoves, 11, "minimum number of game moves to be included in filtered dataset");
ABSL_FLAG(int, minTime, 30, "minimum time remaining to be included in filtered dataset");
ABSL_FLAG(std::string, outdir, "", "output directory for writing memmap files");
ABSL_FLAG(float, trainp, 0.9, "percentage of dataset for training");
ABSL_FLAG(float, testp, 0.08, "percentage of dataset for testing");
ABSL_FLAG(std::vector<std::string>, eloEdges, std::vector<std::string>({"1000","1200","1400","1600","1800","2000","2200","2400","2600"}), "ELO bin edges for ensuring even distribution of ELOs");
ABSL_FLAG(int, maxGamesPerElo, 5000000, "maximum number of games per ELO group");

MMCRawDataReader::MMCRawDataReader(std::string npydir) {
	gamestarts = std::ifstream(npydir + "/gamestarts.npy", std::ios::binary);
	eloWhite = std::ifstream(npydir + "/welos.npy", std::ios::binary);
	eloBlack = std::ifstream(npydir + "/belos.npy", std::ios::binary);

	gamestarts.seekg(0, gamestarts.end);
	size_t nbytes = gamestarts.tellg();
	totalGames = nbytes/8;
	gamestarts.seekg(0, gamestarts.beg);

	gamestarts.read((char*)&gameStart, sizeof(gameStart));
	eloWhite.read((char*)&whiteElo, sizeof(whiteElo));
	eloBlack.read((char*)&blackElo, sizeof(blackElo));

	clktimes = std::ifstream(npydir + "/clk.npy", std::ios::binary);
}

int64_t MMCRawDataReader::getTotalGames() {
	return totalGames;
}

std::tuple<size_t, size_t, int16_t, int16_t> MMCRawDataReader::nextGame(std::vector<int16_t>& clkVec) {
	
	clkVec.clear();

	int64_t gameEnd;	
	size_t gameSize;

	while (true) {
		gamestarts.read((char*)&gameEnd, sizeof(gameEnd));
		if (gamestarts.gcount() <= 0 || gameEnd == 0) {
			return std::make_tuple(0,0,0,0);
		}

		gameSize = gameEnd-gameStart;
		if (gameSize > 0) break;

	}

	size_t nbytes = gameSize*sizeof(int16_t);
	int16_t* buf = (int16_t*)malloc(nbytes);

	clktimes.read((char*)buf, nbytes);
	clkVec.reserve(gameSize);
	clkVec.insert(clkVec.begin(), buf, &buf[gameSize]);

	free(buf);
	size_t gs = gameStart;
	gameStart = gameEnd;
	int16_t we = whiteElo;	
	int16_t be = blackElo;

	eloWhite.read((char*)&whiteElo, sizeof(whiteElo));
	eloBlack.read((char*)&blackElo, sizeof(blackElo));

	return std::make_tuple(nbytes, gs, we, be);
}

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
	Split(): nSamp(0), nGames(0) {};
};



void filterData(std::string& npydir, int minMoves, int minTime, std::string& outdir, float trainp, float testp, std::vector<int>& eloEdges, int maxGames) {
	MMCRawDataReader mrd(npydir);
	std::vector<int16_t> clk;
	std::ofstream gsfile(outdir + "/gs.npy", std::ios::binary);
	std::ofstream elofile(outdir + "/elo.npy", std::ios::binary);
	std::vector<Split> splits(3);
	std::vector<std::string> names = {"train", "test", "val"};
	std::vector<int> eloHist(eloEdges.size()+1, 0);
	for (int i=0; i<3; i++) {
		splits[i].name = names[i];
		splits[i].idxData = std::ofstream(outdir + "/" + names[i] + ".npy", std::ios::binary);
	}
	RandDS rds(trainp, testp);

	int64_t nGames = 0;
	auto insertCoords = [&](int64_t gStart, int64_t gLength, int16_t welo, int16_t belo){
		int dsIdx = rds.get();
		
		splits[dsIdx].idxData.write((char*)&gStart, sizeof(int64_t));
		splits[dsIdx].idxData.write((char*)&gLength, sizeof(int64_t));
		splits[dsIdx].idxData.write((char*)&nGames, sizeof(int64_t));
		splits[dsIdx].nGames++;
	
		gsfile.write((char*)&gStart, sizeof(gStart));
		elofile.write((char*)&welo, sizeof(welo));
		elofile.write((char*)&belo, sizeof(belo));

		nGames++;
	};
	auto getEloBin = [&](int elo) {
		for (int i=0; i<eloEdges.size(); i++) {
			if (eloEdges[i] > elo) {
				return i;
			}
		}
		return (int)eloEdges.size();
	};
	
	int nTotal = 0;
	while (true) {
	 	auto [bytesRead, gameStart, whiteElo, blackElo] = mrd.nextGame(clk);
		if (bytesRead == 0) break;

		nTotal++;
		if (nTotal % 1000 == 0) {
			std::cout << int(100*(float)nTotal/mrd.getTotalGames()) << "% done\r";
		}

		int wbin = getEloBin(whiteElo);
		int bbin = getEloBin(blackElo);
		if (eloHist[wbin] == maxGames || eloHist[bbin] == maxGames) continue;
		eloHist[wbin]++;
		if (bbin != wbin) eloHist[bbin]++;
		
		int idx = clk.size()-1;	
		while (idx >= minMoves && clk[idx] < minTime && clk[idx-1] < minTime) idx--;
		if (idx >= minMoves) {
			insertCoords(gameStart, idx+1, whiteElo, blackElo);
		}

		bool done = true;
		for (auto count: eloHist) {
			done = done && count >= maxGames;
		}
		if (done) {
			std::cout << std::endl << "Reached max games; terminating early" << std::endl;
			break;
		}
	}	
	std::cout << "Included " << nGames << " out of " << nTotal << " games" << std::endl;
	for (int i=0; i<eloEdges.size(); i++) std::cout << "Elo <" << eloEdges[i] << ": " << eloHist[i] << std::endl;	
	std::cout << "Elo >" << eloEdges[eloEdges.size()-1] << ": " << eloHist[eloEdges.size()] << std::endl;

	gsfile.close();
	elofile.close();

	json md;
	md["ngames"] = nGames;
	md["min_moves"] = minMoves;
	
	for (int i=0; i<splits.size(); i++) {
		Split& split = splits[i];
		split.idxData.close();
		md[split.name + "_shape"] = {split.nGames,3};
		md[split.name + "_n"] = split.nGames;
	}
	std::ofstream mdfile(outdir + "/fmd.json");
	mdfile << md << std::endl;
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
	int maxGames = absl::GetFlag(FLAGS_maxGamesPerElo);
	filterData(npydir, minMoves, minTime, outdir, trainp, testp, eloEdges, maxGames);
	return 0;
}
