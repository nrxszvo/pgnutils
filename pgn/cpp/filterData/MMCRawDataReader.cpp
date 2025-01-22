#include "MMCRawDataReader.h"
#include <re2/re2.h>

MMCRawDataReader::MMCRawDataReader(std::string npydir, int64_t startGame, int64_t nGamesToProcess): nGames(0) {
	gsIf = std::ifstream(npydir + "/gamestarts.npy", std::ios::binary);
	weloIf = std::ifstream(npydir + "/welos.npy", std::ios::binary);
	beloIf = std::ifstream(npydir + "/belos.npy", std::ios::binary);

	int64_t startByte = startGame*sizeof(int64_t);

	gsIf.seekg(startByte, gsIf.beg);
	weloIf.seekg(startByte, weloIf.beg);
	beloIf.seekg(startByte, beloIf.beg);

	if (nGamesToProcess == -1) {
		gsIf.seekg(startByte, gsIf.end);
		size_t nbytes = gsIf.tellg();
		totalGames = nbytes/sizeof(int64_t);
		gsIf.seekg(startByte, gsIf.beg);
	} else {
		totalGames = nGamesToProcess;
	}
	
	gsIf.read((char*)&gameStart, sizeof(gameStart));
	weloIf.read((char*)&whiteElo, sizeof(whiteElo));
	beloIf.read((char*)&blackElo, sizeof(blackElo));

	clkIf = std::ifstream(npydir + "/clk.npy", std::ios::binary);
}

int64_t MMCRawDataReader::getTotalGames() {
	return totalGames;
}

std::tuple<size_t, size_t, int16_t, int16_t> MMCRawDataReader::nextGame(std::vector<int16_t>& clkVec) {
	
	clkVec.clear();

	if (nGames++ == totalGames) return std::make_tuple(0,0,0,0);

	int64_t gameEnd;	
	size_t gameSize;

	while (true) {
		gsIf.read((char*)&gameEnd, sizeof(gameEnd));
		if (gsIf.gcount() <= 0 || gameEnd == 0) {
			return std::make_tuple(0,0,0,0);
		}

		gameSize = gameEnd-gameStart;
		if (gameSize > 0) break;

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

	weloIf.read((char*)&whiteElo, sizeof(whiteElo));
	beloIf.read((char*)&blackElo, sizeof(blackElo));

	return std::make_tuple(nbytes, gs, we, be);
}


