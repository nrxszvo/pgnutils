#include "parallelParser.h"
#include "lib/decompress.h"
#include "lib/parseMoves.h"
#include "lib/validate.h"
#include "lib/utils.h"
#include <fstream>
#include <stdexcept>
#include <thread>
#include <filesystem>
#include <chrono>
#include <iostream>

namespace fs = std::filesystem;

struct GameState {
	std::queue<std::string> *pgnQ;
	std::queue<std::shared_ptr<GameData> >* gamesQ;
	std::mutex* pgnMtx;
	std::mutex* gamesMtx;
	std::condition_variable* pgnCv;
	std::condition_variable* gamesCv;
	GameState(
			std::queue<std::string>* pgnQ, 
			std::queue<std::shared_ptr<GameData> >* gamesQ, 
			std::mutex* pgnMtx,
			std::mutex* gamesMtx, 
			std::condition_variable* pgnCv,
			std::condition_variable* gamesCv) 
		: pgnQ(pgnQ), gamesQ(gamesQ), pgnMtx(pgnMtx), gamesMtx(gamesMtx), pgnCv(pgnCv), gamesCv(gamesCv) {};

};

void loadGamesZst(GameState gs, int nReaders) {
	while(true) {
		std::string zst;
		{
			std::unique_lock<std::mutex> lock(*gs.pgnMtx);
			gs.pgnCv->wait(lock, [&] { return !gs.pgnQ->empty(); });
			zst = gs.pgnQ->front();
			gs.pgnQ->pop();
			lock.unlock();
		}
		if (zst == "DONE") {
			{
				std::lock_guard<std::mutex> lock(*gs.gamesMtx);
				for (int i=0; i<nReaders; i++) {
					gs.gamesQ->push(std::make_shared<GameData>("SESSION_DONE"));
				}
			}
			gs.gamesCv->notify_all();
			break;
		} else {
			uintmax_t bytesProcessed = 0;
			int gameId = 0;
			int gamestart = 0;
			int lineno = 0;
			PgnProcessor processor;
			DecompressStream decompressor(zst);
			while(decompressor.decompressFrame() != 0) {
				std::vector<std::string> lines = decompressor.getOutput();
				bytesProcessed += decompressor.getFrameSize();
				for (auto line: lines) {
					lineno++;
					std::string code = processor.processLine(line);
					if (code == "COMPLETE") {
						{
							std::lock_guard<std::mutex> lock(*gs.gamesMtx);
							gs.gamesQ->push(std::make_shared<GameData>(
									bytesProcessed, 
									gameId, 
									processor.getWelo(), 
									processor.getBelo(), 
									processor.getMoveStr(),
									zst + ":" + std::to_string(gamestart)
								));
						}
						gs.gamesCv->notify_one();
						gamestart = lineno + 1;
						gameId++;
					}
				}
			}	
			{
				std::lock_guard<std::mutex> lock(*gs.gamesMtx);
				for (int i=0; i<nReaders; i++) {
					gs.gamesQ->push(std::make_shared<GameData>("FILE_DONE", gameId));
				}
			}
			gs.gamesCv->notify_all();
		}
	}
}

void loadGames(GameState gs, int nReaders) {
	while(true) {
		std::string pgn;
		{
			std::unique_lock<std::mutex> lock(*gs.pgnMtx);
			gs.pgnCv->wait(lock, [&] { return !gs.pgnQ->empty(); });
			pgn = gs.pgnQ->front();
			gs.pgnQ->pop();
			lock.unlock();
		}
		if (pgn == "DONE") {
			{
				std::lock_guard<std::mutex> lock(*gs.gamesMtx);
				for (int i=0; i<nReaders; i++) {
					gs.gamesQ->push(std::make_shared<GameData>("SESSION_DONE"));
				}
			}
			gs.gamesCv->notify_all();
			break;
		} else {
			uintmax_t bytesProcessed = 0;
			int gameId = 0;
			int gamestart = 0;
			int lineno = 0;
			std::ifstream infile(pgn);
			std::string line;
			PgnProcessor processor;
			while (std::getline(infile, line)) {
				lineno++;
				bytesProcessed += line.size();
				std::string code = processor.processLine(line);
				if (code == "COMPLETE") {
					{
						std::lock_guard<std::mutex> lock(*gs.gamesMtx);
						gs.gamesQ->push(std::make_shared<GameData>(
								bytesProcessed, 
								gameId, 
								processor.getWelo(), 
								processor.getBelo(), 
								processor.getMoveStr(),
								pgn + ":" + std::to_string(gamestart)
							));
					}
					gs.gamesCv->notify_one();
					gamestart = lineno + 1;
					gameId++;
				}
			}	
			{
				std::lock_guard<std::mutex> lock(*gs.gamesMtx);
				for (int i=0; i<nReaders; i++) {
					gs.gamesQ->push(std::make_shared<GameData>("FILE_DONE", gameId));
				}
			}
			gs.gamesCv->notify_all();
		}
	}
}

std::shared_ptr<std::thread> startGamesReader(
		std::queue<std::string>& pgnQ, 
		std::queue<std::shared_ptr<GameData> >& gamesQ, 
		std::mutex& pgnMtx,
		std::mutex& gamesMtx,
		std::condition_variable& pgnCv,
	   	std::condition_variable& gamesCv,
	   	int nReaders) {
	GameState gs(&pgnQ, &gamesQ, &pgnMtx, &gamesMtx, &pgnCv, &gamesCv);
	return std::make_shared<std::thread>(loadGamesZst, gs, nReaders);
}

struct ReaderState {
	std::queue<std::shared_ptr<GameData> >* gamesQ;
	std::queue<std::shared_ptr<MoveData> >* outputQ;
	int pid;
	std::mutex* gamesMtx;
	std::mutex* outputMtx;
	std::condition_variable* gamesCv;
	std::condition_variable* outputCv;
	ReaderState(std::queue<std::shared_ptr<GameData> >* gamesQ, std::queue<std::shared_ptr<MoveData> >* outputQ, std::mutex* gamesMtx, std::mutex* outputMtx, std::condition_variable* gamesCv, std::condition_variable* outputCv) 
		: gamesQ(gamesQ), outputQ(outputQ), pid(-1), gamesMtx(gamesMtx), outputMtx(outputMtx), gamesCv(gamesCv), outputCv(outputCv) {};
};


void processGames(ReaderState ts, bool requireClk) {		
	while(true) {
		std::shared_ptr<GameData> gd;
		{
			std::unique_lock<std::mutex> lock(*ts.gamesMtx);
			ts.gamesCv->wait(lock, [&]{return !ts.gamesQ->empty();});
			gd = ts.gamesQ->front();
			ts.gamesQ->pop();
			lock.unlock();
		}
		if (gd->info == "FILE_DONE") {
			std::lock_guard<std::mutex> lock(*ts.outputMtx);
			ts.outputQ->push(std::make_shared<MoveData>("DONE", gd->gameId));
			ts.outputCv->notify_one();
		} else if (gd->info == "SESSION_DONE") {
			std::lock_guard<std::mutex> lock(*ts.outputMtx);
			ts.outputQ->push(std::make_shared<MoveData>("SESSION_DONE"));
			ts.outputCv->notify_one();
			break;
		} else {
			try {
				auto [mvids, clk] = parseMoves(gd->moveStr, requireClk);
				auto errs = validateGame(gd->gameId, gd->moveStr, mvids);
				{
					std::lock_guard<std::mutex> lock(*ts.outputMtx);
					if (errs.empty()) {
						ts.outputQ->push(std::make_shared<MoveData>(ts.pid, gd, mvids, clk));
					} else {
						ts.outputQ->push(std::make_shared<MoveData>(errs));
					}
					ts.outputCv->notify_one();
				}
			} catch(std::exception &e) {
				std::lock_guard<std::mutex> lock(*ts.outputMtx);
				ts.outputQ->push(std::make_shared<MoveData>("INVALID"));
			}
		}	
	}
}

std::vector<std::shared_ptr<std::thread> >  startReaderThreads(
		int nReaders, 
		std::queue<std::shared_ptr<GameData> >& gamesQ, 
		std::queue<std::shared_ptr<MoveData> >& outputQ, 
		std::mutex& gamesMtx, 
		std::mutex& outputMtx,
	   	std::condition_variable& gamesCv, 
		std::condition_variable& outputCv,
		bool requireClk) {

	std::vector<std::shared_ptr<std::thread> > threads;
	ReaderState ts(&gamesQ, &outputQ, &gamesMtx, &outputMtx, &gamesCv, &outputCv);
	for (int i=0; i<nReaders; i++) {
		ts.pid = i;
		threads.push_back(std::make_shared<std::thread>(processGames, ts, requireClk));
	}	
	return threads;
}

ParallelParser::ParallelParser(int nReaders, bool requireClk) : nReaders(nReaders) {
	this->readerThreads = startReaderThreads(
		nReaders, 
		this->gamesQ, 
		this->outputQ, 
		this->gamesMtx,
		this->outputMtx,
		this->gamesCv,
		this->outputCv,
		requireClk
	);
	this->gameThread = startGamesReader(
		this->pgnQ, 
		this->gamesQ, 
		this->pgnMtx,
		this->gamesMtx, 
		this->pgnCv,
		this->gamesCv, 
		nReaders
	);
};

ParallelParser::~ParallelParser() {
	{
		std::lock_guard<std::mutex> lock(this->pgnMtx);
		this->pgnQ.push("DONE");
		this->pgnCv.notify_one();
	}
	{
		std::unique_lock<std::mutex> lock(this->outputMtx);
		this->outputCv.wait(lock, [&]{return this->outputQ.size()==this->nReaders;});
		for (int i=0; i<this->nReaders; i++) {
			this->outputQ.pop();
		}
		lock.unlock();
	}

	this->gameThread->join();
	for (auto rt: this->readerThreads) {
		rt->join();
	}
}

ParserOutput ParallelParser::parse(std::string pgn, std::string name, int printFreq) {
	uintmax_t nbytes = fs::file_size(pgn);	
	{
		std::unique_lock<std::mutex> lock(this->pgnMtx);
		this->pgnQ.push(pgn);
		this->pgnCv.notify_one();
		lock.unlock();
	}

	auto welos = std::make_shared<std::vector<int16_t> >();
	auto belos = std::make_shared<std::vector<int16_t> >();
	auto gamestarts = std::make_shared<std::vector<int64_t> >();
	auto mvids = std::make_shared<std::vector<int16_t> >();
	auto clktimes = std::make_shared<std::vector<int16_t> >();

	int64_t ngames = 0;
	int totalGames = INT_MAX;
	int nFinished = 0;
	size_t maxBytes = 0;
	auto start = hrc::now();
	auto lastPrintTime = start;
	while (ngames < totalGames || nFinished < this->nReaders) {
		std::shared_ptr<MoveData> md;
		{
			std::unique_lock<std::mutex> lock(this->outputMtx);
			this->outputCv.wait(lock, [&]{return !this->outputQ.empty();});
			md = this->outputQ.front();
			this->outputQ.pop();
			lock.unlock();
		}
		if (md->info == "DONE") {
			totalGames = md->gameId;
			nFinished++;
		} else if (md->info == "ERROR") {
			ngames++;
		} else if (md->info == "INVALID") {
			ngames++;
		} else if (md->info == "GAME") {
			maxBytes = std::max(maxBytes, md->bytesProcessed);	
			welos->push_back(md->welo);
			belos->push_back(md->belo);
			gamestarts->push_back(mvids->size());
			mvids->insert(mvids->end(), md->mvids.begin(), md->mvids.end());
			clktimes->insert(clktimes->end(), md->clk.begin(), md->clk.end());
			ngames++;
			
			int totalGamesEst = ngames / ((float)maxBytes / nbytes);
			int curProg = int(100.0f * ngames / totalGamesEst);
			if (ellapsedGTE(lastPrintTime, printFreq)) {
				auto [eta, ellapsed] = getEta(totalGamesEst, ngames, start);
				int gamesPerSec = 1000*ngames/ellapsed;
				int mbps = 1000*maxBytes/ellapsed/(1024*1024);
				std::string status = name + ": parsed " + std::to_string(ngames) + \
									 " games (" + std::to_string(curProg) + \
									 "% done, MB/sec: " + std::to_string(mbps) + \
									 ", games/sec: " + std::to_string(gamesPerSec) + \
									 ", eta: " + eta + ")";
				std::cout << status << std::endl;
				lastPrintTime = hrc::now();
			}
		} else {
			throw std::runtime_error("invalid code: " + md->info);
		}
	}
	return ParserOutput(welos, belos, gamestarts, mvids, clktimes);
}
