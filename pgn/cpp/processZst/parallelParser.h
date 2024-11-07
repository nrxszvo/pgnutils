#include <queue>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <thread>
#include "parser.h"

struct Data {
	int pid;
	float progress;
	int gameId;
	int welo;
	int belo;
	std::string info; 

	Data() {};
	Data(int pid, std::string info): pid(pid), info(info) {};
	Data(int pid, std::string info, int gameId): pid(pid), info(info), gameId(gameId) {};
	Data(int pid, float progress, int gid, int welo, int belo, std::string info)
		: pid(pid), progress(progress), gameId(gid), welo(welo), belo(belo), info(info) {};
	Data(int pid, std::shared_ptr<Data> other)
		: pid(pid), progress(other->progress), gameId(other->gameId), welo(other->welo), belo(other->belo) {};
};

struct GameData: Data {
	GameData(): Data() {};
	GameData(int pid, std::string info) : Data(pid, info) {};
	GameData(int pid, std::string info, int gameId): Data(pid, info, gameId) {};
	GameData(int pid, float progress, int gid, int welo, int belo, std::string moveStr, std::string info) 
		: Data(pid, progress, gid, welo, belo, info), moveStr(moveStr) {};

	std::string moveStr;
};

struct MoveData: Data {
	MoveData(int pid, std::string info) : Data(pid, info) {};
	MoveData(int pid, std::string info, int gameId): Data(pid, info, gameId) {};
	MoveData(
			int pid, 
			std::shared_ptr<GameData> gd, 
			std::vector<int16_t>& mvids, 
			std::vector<int16_t>& clk
			) 
		: Data(pid, gd), mvids(mvids), clk(clk) {
		this->info = "GAME";
	};
	MoveData(std::vector<std::pair<int,std::string> >& errs) 
		: errs(errs) { 
		this->info = "ERROR"; 
	};

	std::vector<int16_t> mvids;
	std::vector<int16_t> clk;
	std::vector<std::pair<int, std::string> > errs;
};

class ParallelParser {
	std::queue<std::shared_ptr<GameData> > gamesQ;
	std::queue<std::shared_ptr<MoveData> > outputQ;
	std::mutex gamesMtx;
	std::mutex outputMtx;
	std::condition_variable gamesCv;
	std::condition_variable outputCv;
	int nReaders;
	int nMoveProcessors;
	std::vector<std::shared_ptr<std::thread> > procThreads;
	std::vector<std::shared_ptr<std::thread> > gameThreads;
public:
	ParallelParser(int nReaders, int nMoveProcessors);
	~ParallelParser();
	ParserOutput parse(std::string zst, std::string name, bool requireClk, int printFreq=60);
};
