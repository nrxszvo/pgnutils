#include "MMCRawDataReader.h"
#include "npy.hpp"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include <fstream>
#include <iostream>
#include <random>

ABSL_FLAG(std::string, npydir, "", "directory containing npy raw data files");
ABSL_FLAG(int, minMoves, 11, "minimum number of game moves to be included in filtered dataset");
ABSL_FLAG(int, minTime, 30, "minimum time remaining to be included in filtered dataset");
ABSL_FLAG(std::string, outdir, "", "output directory for writing memmap files");
ABSL_FLAG(float, trainp, 0.8, "percentage of dataset for training");
ABSL_FLAG(float, testp, 0.1, "percentage of dataset for testing");

MMCRawDataReader::MMCRawDataReader(std::string npydir) {
	gamestarts = std::ifstream(npydir + "/gamestarts.npy", std::ios::binary);
	eloWhite = std::ifstream(npydir + "/welos.npy", std::ios::binary);
	eloBlack = std::ifstream(npydir + "/belos.npy", std::ios::binary);

	gamestarts.read((char*)&gameStart, sizeof(gameStart));
	eloWhite.read((char*)&whiteElo, sizeof(whiteElo));
	eloBlack.read((char*)&blackElo, sizeof(blackElo));

	clktimes = std::ifstream(npydir + "/clk.npy", std::ios::binary);
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
	int64_t cum;
	std::vector<int64_t> idxData;
	Split(): cum(0) {};
};

void filterData(std::string& npydir, int minMoves, int minTime, std::string& outdir, float trainp, float testp) {
	MMCRawDataReader mrd(npydir);
	std::vector<int16_t> clk;
	std::vector<int64_t> gsvec;
	std::vector<int16_t> elovec;

	std::vector<Split> splits(3);
	RandDS rds(trainp, testp);

	auto insertCoords = [&](size_t gs, int idx, int welo, int belo){
		int ds_idx = rds.get();
		size_t nsamp = idx+1-gs-minMoves;
		splits[ds_idx].idxData.push_back(splits[ds_idx].cum);
		splits[ds_idx].cum += nsamp;
		splits[ds_idx].idxData.push_back(gsvec.size());
		gsvec.push_back(gs);
		elovec.push_back(welo);
		elovec.push_back(belo);
	};

	int nTotal = 0;
	int nInc = 0;
	while (true) {
	 	auto [bytesRead, gameStart, whiteElo, blackElo] = mrd.nextGame(clk);
		if (bytesRead == 0) break;
		nTotal++;
		int idx = clk.size()-1;	
		while (idx >= minMoves && clk[idx] < minTime && clk[idx-1] < minTime) idx--;
		if (idx >= minMoves) {
			nInc++;
			insertCoords(gameStart, idx, whiteElo, blackElo);
		}
	}	
	std::cout << "Included " << nInc << " out of " << nTotal << " games" << std::endl;
	
	npy::npy_data_ptr<int64_t> gsnpy;
	gsnpy.data_ptr = gsvec.data();
	gsnpy.shape = { gsvec.size() };
	npy::write_npy(outdir + "/gs.npy", gsnpy);

	npy::npy_data_ptr<int16_t> elonpy;
	elonpy.data_ptr = elovec.data();
	elonpy.shape = { elovec.size()/2, 2 };
	npy::write_npy(outdir + "/elo.npy", elonpy);

	for (int i=0; i<splits.size(); i++) {
		std::string name;
		if (i == 0) name = "train.npy";
		else if (i == 1) name = "test.npy";
		else name = "val.npy";
		Split& split = splits[i];

		npy::npy_data_ptr<int64_t> idnpy;
		idnpy.data_ptr = split.idxData.data();
		idnpy.shape = { split.idxData.size()/2, 2 };
		npy::write_npy(outdir + "/" + name, idnpy);
	}
}


int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("filter raw MimicChess dataset based on minimum number-of-moves and time-remaining constraints");
	absl::ParseCommandLine(argc, argv);
	std::string npydir = absl::GetFlag(FLAGS_npydir);
	int minMoves = absl::GetFlag(FLAGS_minMoves);
	int minTime = absl::GetFlag(FLAGS_minTime);
	std::string outdir = absl::GetFlag(FLAGS_outdir);
	float trainp = absl::GetFlag(FLAGS_trainp);
	float testp = absl::GetFlag(FLAGS_testp);
	filterData(npydir, minMoves, minTime, outdir, trainp, testp);
	return 0;
}
