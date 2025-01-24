#include <vector>
#include <string>
#include <fstream>
#include <vector>
#include <tuple>

class MMCRawDataReader {		
	std::ifstream gsIf;
	std::ifstream clkIf;
	std::ifstream weloIf;
	std::ifstream beloIf;
	std::string npydir;
	int64_t totalGames;
	int64_t startGame;
	int64_t nGames;
	int64_t gameStart;
	int16_t whiteElo;
	int16_t blackElo;
public:
	MMCRawDataReader(std::string npydir, int64_t startGame=0, int64_t nGamesToProcess =-1);
	std::tuple<size_t,size_t,size_t,int16_t,int16_t> nextGame(std::vector<int16_t>& clk);
	int64_t getTotalGames();
};
