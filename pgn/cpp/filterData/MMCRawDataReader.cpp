#include "MMCRawDataReader.h"
#include <re2/re2.h>
#include <stdexcept>

MMCRawDataReader::MMCRawDataReader(std::string npydir, int64_t startGame, int64_t nGamesToProcess): npydir(npydir), startGame(startGame), nGames(0) {
	gsIf = std::ifstream(npydir + "/gamestarts.npy", std::ios::binary);
	weloIf = std::ifstream(npydir + "/welos.npy", std::ios::binary);
	beloIf = std::ifstream(npydir + "/belos.npy", std::ios::binary);

	int64_t startByte64 = startGame*sizeof(int64_t);
	int64_t startByte16 = startGame*sizeof(int16_t);

	weloIf.seekg(startByte16, weloIf.beg);
	beloIf.seekg(startByte16, beloIf.beg);

	if (nGamesToProcess == -1) {
		gsIf.seekg(0, gsIf.end);
		size_t nbytes = gsIf.tellg()-startByte64;
		totalGames = nbytes/sizeof(int64_t);
	} else {
		totalGames = nGamesToProcess;
	}
	
	gsIf.seekg(startByte64, gsIf.beg);

	gsIf.read((char*)&gameStart, sizeof(gameStart));
	weloIf.read((char*)&whiteElo, sizeof(whiteElo));
	beloIf.read((char*)&blackElo, sizeof(blackElo));

	clkIf = std::ifstream(npydir + "/clk.npy", std::ios::binary);
	clkIf.seekg(gameStart*sizeof(int16_t), clkIf.beg);
}

int64_t MMCRawDataReader::getTotalGames() {
	return totalGames;
}

std::tuple<size_t, size_t, size_t, int16_t, int16_t> MMCRawDataReader::nextGame(std::vector<int16_t>& clkVec) {
	
	clkVec.clear();

	if (nGames == totalGames) return std::make_tuple(0,0,0,0,0);

	int64_t gameEnd;	
	size_t gameSize;

	while (true) {
		gsIf.read((char*)&gameEnd, sizeof(gameEnd));
		if (gameEnd == 0) throw std::runtime_error("gameEnd is 0");
		if (gsIf.gcount() <= 0) {
			int64_t pos = clkIf.tellg();
			clkIf.seekg(0, clkIf.end);
			int64_t rem = clkIf.tellg() - pos;
			clkIf.seekg(pos, clkIf.beg);
			gameSize = rem/sizeof(int64_t);
			if (nGames != totalGames-1) throw std::runtime_error("permaturely reached eof (nGames=" + std::to_string(nGames) + ", total games=" + std::to_string(totalGames) + ") in " + npydir);
		} else {
			gameSize = gameEnd-gameStart;
		}
		if (gameSize > 0) break;
		totalGames--;
	}

	size_t nbytes = gameSize*sizeof(int16_t);
	int16_t* buf = (int16_t*)malloc(nbytes);

	clkIf.read((char*)buf, nbytes);
	clkVec.reserve(gameSize);
	clkVec.insert(clkVec.begin(), buf, &buf[gameSize]);

	free(buf);
	size_t gs = gameStart;
	gameStart = gameEnd;
	int16_t we = whiteElo;	
	int16_t be = blackElo;
	size_t gIdx = startGame + nGames;
	nGames++;

	weloIf.read((char*)&whiteElo, sizeof(whiteElo));
	beloIf.read((char*)&blackElo, sizeof(blackElo));

	return std::make_tuple(nbytes, gIdx, gs, we, be);
}


