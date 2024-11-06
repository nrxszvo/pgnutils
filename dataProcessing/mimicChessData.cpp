#include "MMCRawDataReader.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include <filesystem>

ABSL_FLAG(std::string, npydir, "", "directory containing npy raw data files");
ABSL_FLAG(int, minMoves, 11, "minimum number of game moves to be included in filtered dataset");
ABSL_FLAG(int, minTime, 30, "minimum time remaining to be included in filtered dataset");
ABSL_FLAG(std::string, outdir, "", "output directory for writing memmap files");

MMCRawDataReader::MMCRawDataReader(std::string npydir) {
	gamestarts = std::ifstream(npydir + "/gamestarts.npy", std::ios::binary);
	eloWhite = std::ifstream(npydir + "/elos.npy", std::ios::binary);
	auto nbytes = std::filesystem::file_size(npydir + "/elos.npy");
	eloBlack = std::ifstream(npydir + "/elos.npy", std::ios::binary);
	eloBlack.seekg(nbytes/2, eloBlack.beg);

	gamestarts.read((char*)&gameStart, sizeof(gameStart));
	eloWhite.read((char*)&whiteElo, sizeof(whiteElo));
	eloBlack.read((char*)&blackElo, sizeof(blackElo));

	mvids = std::ifstream(npydir + "/moves.npy", std::ios::binary);

	nbytes = std::filesystem::file_size(npydir + "/moves.npy");
	clktimes = std::ifstream(npydir + "/moves.npy", std::ios::binary);
	clktimes.seekg(nbytes/2, clktimes.beg);

	}

std::tuple<size_t, size_t, int16_t, int16_t> MMCRawDataReader::nextGame(std::vector<int16_t>& mvidVec, std::vector<int16_t>& clkVec) {
	
	mvidVec.clear();
	clkVec.clear();

	int64_t gameEnd;	
	gamestarts.read((char*)&gameEnd, sizeof(gameEnd));
	if (gamestarts.gcount() <= 0) {
		return std::make_tuple(0,0,0,0);
	}

	size_t gameSize = gameEnd-gameStart;

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

int getBin(int elo, std::vector<int>& binEdges) {
	for (int i=0; i<binEdges.size(); i++) {
		if (elo < binEdges[i]) {
			return i;
		}
	}
	return binEdges.size();
}

int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("filter raw MimicChess dataset based on minimum number-of-moves and time-remaining constraints");
	absl::ParseCommandLine(argc, argv);

	MMCRawDataReader mrd(absl::GetFlag(FLAGS_npydir));
	int minMoves = absl::GetFlag(FLAGS_minMoves);
	int minTime = absl::GetFlag(FLAGS_minTime);
	std::string outdir = absl::GetFlag(FLAGS_outdir);

	std::vector<int16_t> mvids;
	std::vector<int16_t> clk;
	std::vector<int> edges = {1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000};
	std::vector<std::vector<std::pair<int64_t,int64_t> > > filteredWhite(edges.size()+1);
	std::vector<std::vector<std::pair<int64_t,int64_t> > > filteredBlack(edges.size()+1);

	auto insertCoords = [&](size_t gs, int idx, int welo, int belo){
		size_t ge = gs+idx-1;
		auto coords = std::make_pair(gs, ge);
		if (idx % 2 == 0) {
			int bin = getBin(welo, edges);
			filteredWhite[bin].push_back(coords);
		} else {
			int bin = getBin(belo, edges);
			filteredBlack[bin].push_back(coords);	
		}
	};

	while (true) {
	 	auto [bytesRead, gameStart, whiteElo, blackElo] = mrd.nextGame(mvids, clk);
		if (bytesRead == 0) break;

		int idx = mvids.size()-1;	
		while (idx >= minMoves && clk[idx] < minTime) idx--;
		if (idx >= minMoves) {
			insertCoords(gameStart, idx, whiteElo, blackElo);
			idx--;
			while (idx >= minMoves && clk[idx] < minTime) idx -= 2;
			if (idx >= minMoves) {
				insertCoords(gameStart, idx, whiteElo, blackElo); 
			}
		}
	}	
	auto writeData = [&](std::vector<std::pair<int64_t,int64_t> >& filtered, std::string name, int e){
		if (filtered.size() > 0) {
			std::ofstream out(outdir + "/" + name + "-" + std::to_string(e) + ".npy", std::ios::binary);
			out.write((char*)filtered.data(), filtered.size()*sizeof(filtered[0]));
		}
	};	
	for (int i=0; i<edges.size(); i++) {
		writeData(filteredWhite[i], "white", edges[i]);	
		writeData(filteredBlack[i], "black", edges[i]);
	}
	return 0;
}


