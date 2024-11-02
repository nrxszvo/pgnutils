#include <queue>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <thread>
#include "parser.h"

struct Data {
	uintmax_t bytesProcessed;
	int gameId;
	int welo;
	int belo;
	std::string info; 

	Data() {};
	Data(std::string info): info(info) {};
	Data(std::string info, int gameId): info(info), gameId(gameId) {};
	Data(uintmax_t bp, int gid, int welo, int belo, std::string info)
		: bytesProcessed(bp), gameId(gid), welo(welo), belo(belo), info(info) {};
	Data(std::shared_ptr<Data> other)
		: bytesProcessed(other->bytesProcessed), gameId(other->gameId), welo(other->welo), belo(other->belo) {};
};

struct GameData: Data {
	GameData(): Data() {};
	GameData(std::string info) : Data(info) {};
	GameData(std::string info, int gameId): Data(info, gameId) {};
	GameData(uintmax_t bp, int gid, int welo, int belo, std::string moveStr, std::string info) 
		: Data(bp, gid, welo, belo, info), moveStr(moveStr) {};

	std::string moveStr;
};

struct MoveData: Data {
	MoveData(std::string info) : Data(info) {};
	MoveData(std::string info, int gameId): Data(info, gameId) {};
	MoveData(
			int pid, 
			std::shared_ptr<GameData> gd, 
			std::vector<int16_t>& mvids, 
			std::vector<int16_t>& clk
			) 
		: pid(pid), Data(gd), mvids(mvids), clk(clk) {
		this->info = "GAME";
	};
	MoveData(std::vector<std::pair<int,std::string> >& errs) 
		: errs(errs) { 
		this->info = "ERROR"; 
	};

	int pid;
	std::vector<int16_t> mvids;
	std::vector<int16_t> clk;
	std::vector<std::pair<int, std::string> > errs;
};

class ParallelParser {
	std::queue<std::string> pgnQ;
	std::queue<std::shared_ptr<GameData> > gamesQ;
	std::queue<std::shared_ptr<MoveData> > outputQ;
	std::mutex pgnMtx;
	std::mutex gamesMtx;
	std::mutex outputMtx;
	std::condition_variable pgnCv;
	std::condition_variable gamesCv;
	std::condition_variable outputCv;
	int nReaders;
	std::vector<std::shared_ptr<std::thread> > readerThreads;
	std::shared_ptr<std::thread> gameThread;
public:
	ParallelParser(int nReaders);
	~ParallelParser();
	ParserOutput parse(std::string pgn, std::string name);
};
