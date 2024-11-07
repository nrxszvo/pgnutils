#include "MMCRawDataReader.h"
#include "npy.hpp"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include <filesystem>
#include <fstream>

ABSL_FLAG(std::string, npydir, "", "directory containing npy raw data files");
ABSL_FLAG(int, minMoves, 11, "minimum number of game moves to be included in filtered dataset");
ABSL_FLAG(int, minTime, 30, "minimum time remaining to be included in filtered dataset");
ABSL_FLAG(std::string, outdir, "", "output directory for writing memmap files");
ABSL_FLAG(std::optional<std::string>, edgesFile, std::nullopt, "text file defining comma-delimited elo bin edges");

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
	gamestarts.read((char*)&gameEnd, sizeof(gameEnd));
	if (gamestarts.gcount() <= 0 || gameEnd == 0) {
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

int qElo(int elo, std::vector<int>& binEdges) {
	for (int i=0; i<binEdges.size(); i++) {
		if (elo < binEdges[i]) {
			return binEdges[i];
		}
	}
	return binEdges[binEdges.size()-1];
}

void filterData(std::string& npydir, int minMoves, int minTime, std::string& outdir, std::vector<int>& edges) {
	MMCRawDataReader mrd(npydir);
	std::vector<int16_t> mvids;
	std::vector<int16_t> clk;
	/*
	std::vector<std::vector<int64_t> > filteredWhite(edges.size()+1);
	std::vector<std::vector<int64_t> > filteredBlack(edges.size()+1);
	*/
	std::vector<int64_t> gsvec;
	std::vector<int64_t> gevec;
	std::vector<int16_t> elovec;

	auto insertCoords = [&](size_t gs, int idx, int welo, int belo){
		size_t ge = gs+idx+1;
		gsvec.push_back(gs);
		gevec.push_back(ge);
		elovec.push_back(welo);
		elovec.push_back(belo);
		
		/*
		if (idx % 2 == 0) {
			int bin = getBin(welo, edges);
			filteredWhite[bin].push_back(gs);
			filteredWhite[bin].push_back(ge);
		} else {
			int bin = getBin(belo, edges);
			filteredBlack[bin].push_back(gs);
			filteredBlack[bin].push_back(ge);	
		}
		*/
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
	/*
	auto writeData = [&](std::vector<int64_t>& filtered, std::string name, int e){
		if (filtered.size() > 0) {
			npy::npy_data_ptr<int64_t> d;
			d.data_ptr = filtered.data();
			d.shape = { filtered.size()/2, 2 };
			npy::write_npy(outdir + "/" + name + "-" + std::to_string(e) + ".npy", d);
		}
	};	
	for (int i=0; i<edges.size(); i++) {
		writeData(filteredWhite[i], "white", edges[i]);	
		writeData(filteredBlack[i], "black", edges[i]);
	}
	*/

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

std::vector<int> readEdgesFile(std::string edgesFn) {
	std::ifstream infile(edgesFn);
	std::string line;
	std::getline(infile, line);
	std::vector<int> edges;
	int idx = 0;
	while (true) {
		int end = line.substr(idx).find(",");
		if (end == std::string::npos) break;
		std::string eloStr = line.substr(idx, idx+end);
		edges.push_back(std::stoi(eloStr));
		idx += end+1;
	}
	return edges;
}

int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("filter raw MimicChess dataset based on minimum number-of-moves and time-remaining constraints");
	absl::ParseCommandLine(argc, argv);
	std::string npydir = absl::GetFlag(FLAGS_npydir);
	int minMoves = absl::GetFlag(FLAGS_minMoves);
	int minTime = absl::GetFlag(FLAGS_minTime);
	std::string outdir = absl::GetFlag(FLAGS_outdir);
	std::vector<int> edges = {1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000};
	if (absl::GetFlag(FLAGS_edgesFile).has_value()) {
		edges = readEdgesFile(absl::GetFlag(FLAGS_edgesFile).value());
	}
	filterData(npydir, minMoves, minTime, outdir, edges);
	return 0;
}
