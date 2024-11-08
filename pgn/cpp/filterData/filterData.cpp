#include "MMCRawDataReader.h"
#include "npy.hpp"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include <fstream>

ABSL_FLAG(std::string, npydir, "", "directory containing npy raw data files");
ABSL_FLAG(int, minMoves, 11, "minimum number of game moves to be included in filtered dataset");
ABSL_FLAG(int, minTime, 30, "minimum time remaining to be included in filtered dataset");
ABSL_FLAG(std::string, outdir, "", "output directory for writing memmap files");

MMCRawDataReader::MMCRawDataReader(std::string npydir) {
	gamestarts = std::ifstream(npydir + "/gamestarts.npy", std::ios::binary);
	eloWhite = std::ifstream(npydir + "/welos.npy", std::ios::binary);
	eloBlack = std::ifstream(npydir + "/belos.npy", std::ios::binary);

	gamestarts.read((char*)&gameStart, sizeof(gameStart));
	eloWhite.read((char*)&whiteElo, sizeof(whiteElo));
	eloBlack.read((char*)&blackElo, sizeof(blackElo));

	mvids = std::ifstream(npydir + "/mvids.npy", std::ios::binary);
	clktimes = std::ifstream(npydir + "/clk.npy", std::ios::binary);
}

std::tuple<size_t, size_t, int16_t, int16_t> MMCRawDataReader::nextGame(std::vector<int16_t>& mvidVec, std::vector<int16_t>& clkVec) {
	
	mvidVec.clear();
	clkVec.clear();

	int64_t gameEnd;	
	size_t gameSize;

	while (true) {
		gamestarts.read((char*)&gameEnd, sizeof(gameEnd));
		if (gamestarts.gcount() <= 0 || gameEnd == 0) {
			std::cout << "gcount: " << gamestarts.gcount() << " gameEnd: " << gameEnd << std::endl;
			return std::make_tuple(0,0,0,0);
		}

		gameSize = gameEnd-gameStart;
		if (gameSize > 0) break;

	}

	size_t nbytes = gameSize*sizeof(int16_t);
	int16_t* buf = (int16_t*)malloc(nbytes);

	mvids.read((char*)buf, nbytes);
	mvidVec.reserve(gameSize);
	mvidVec.insert(mvidVec.begin(), buf, &buf[gameSize]);

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

struct Split {
	int64_t cum;
	std::vector<int64_t> cumSamp;
	std::vector<int64_t> gameStart;
	Split(): cum(0) {};
};
void filterData(std::string& npydir, int minMoves, int minTime, std::string& outdir) {
	MMCRawDataReader mrd(npydir);
	std::vector<int16_t> mvids;
	std::vector<int16_t> clk;
	std::vector<int64_t> gsvec;
	std::vector<int64_t> gevec;
	std::vector<int16_t> elovec;

	std::vector<Split> splits(3);

	auto insertCoords = [&](size_t gs, int idx, int welo, int belo, int ds_idx){
		size_t ge = gs+idx+1;
		splits[ds_idx].cumSamp.push_back(splits[ds_idx].cum);
		splits[ds_idx].cum += ge+1-gs-minMoves;
		splits[ds_idx].gameStart.push_back(gs);

		gsvec.push_back(gs);
		gevec.push_back(ge);
		elovec.push_back(welo);
		elovec.push_back(belo);
		
	};

	int nTotal = 0;
	int nInc = 0;
	while (true) {
	 	auto [bytesRead, gameStart, whiteElo, blackElo] = mrd.nextGame(mvids, clk);
		if (bytesRead == 0) break;
		nTotal++;
		int idx = mvids.size()-1;	
		while (idx >= minMoves && clk[idx] < minTime && clk[idx-1] < minTime) idx--;
		if (idx >= minMoves) {
			nInc++;
			insertCoords(gameStart, idx, whiteElo, blackElo);
			idx--;
		}
	}	
	std::cout << "Included " << nInc << " out of " << nTotal << " games" << std::endl;
	
	npy::npy_data_ptr<int64_t> gsnpy;
	gsnpy.data_ptr = gsvec.data();
	gsnpy.shape = { gsvec.size() };
	npy::write_npy(outdir + "/gs.npy", gsnpy);

	npy::npy_data_ptr<int64_t> genpy;
	genpy.data_ptr = gevec.data();
	genpy.shape = { gevec.size() };
	npy::write_npy(outdir + "/ge.npy", genpy);

	npy::npy_data_ptr<int16_t> elonpy;
	elonpy.data_ptr = elovec.data();
	elonpy.shape = { elovec.size()/2, 2 };
	npy::write_npy(outdir + "/elo.npy", elonpy);
}


int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("filter raw MimicChess dataset based on minimum number-of-moves and time-remaining constraints");
	absl::ParseCommandLine(argc, argv);
	std::string npydir = absl::GetFlag(FLAGS_npydir);
	int minMoves = absl::GetFlag(FLAGS_minMoves);
	int minTime = absl::GetFlag(FLAGS_minTime);
	std::string outdir = absl::GetFlag(FLAGS_outdir);
	filterData(npydir, minMoves, minTime, outdir);
	return 0;
}
