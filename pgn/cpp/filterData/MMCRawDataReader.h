#include <vector>
#include <string>
#include <fstream>
#include <vector>
#include <tuple>

class MMCRawDataReader {		
	std::ifstream gamestarts;
	std::ifstream mvids;
	std::ifstream clktimes;
	std::ifstream eloWhite;
	std::ifstream eloBlack;
	std::ofstream filteredMoves;
	int64_t gameStart;
	int64_t lastGameStart;
	int16_t whiteElo;
	int16_t blackElo;

public:
	MMCRawDataReader(std::string npydir);
	std::tuple<size_t,size_t,int16_t,int16_t> nextGame(std::vector<int16_t>& mvids, std::vector<int16_t>& clk);
};
