#include "lib/parseMoves.h"
#include "lib/decompress.h"
#include "lib/validate.h"
#include "lib/utils.h"
#include "profiling/profiler.h"
#include "serialParser.h"
#include <filesystem>

ParserOutput processSerial(std::string zst) {

	DecompressStream decompressor(zst);
	PgnProcessor processor;

	int gamestart = 0;
	int lineno = 0;

	auto welos = std::make_shared<std::vector<int16_t> >();
	auto belos = std::make_shared<std::vector<int16_t> >();
	auto gamestarts = std::make_shared<std::vector<int64_t> >();
	auto mvids = std::make_shared<std::vector<int16_t> >();
	auto clktimes = std::make_shared<std::vector<int16_t> >();

	uintmax_t nbytes = std::filesystem::file_size(zst);	
	size_t bytesRead;
	uintmax_t bytesProcessed = 0;

	int progress = 0;
	int printFreq = 10;

	profiler.init("parseMoves");
	profiler.init("matchNextMove");
	profiler.init("inferId");
	profiler.init("regex");
	profiler.init("decompress");

	auto start = hrc::now();
	while((bytesRead = decompressor.decompressFrame()) != 0) {
		std::vector<std::string> lines = decompressor.getOutput();
		bytesProcessed += bytesRead;
		for (auto line: lines) {
			lineno++;
			std::string code = processor.processLine(line);		
			if (code == "COMPLETE") {
				profiler.start("parseMoves");
				auto [moves, clk] = parseMoves(processor.getMoveStr());
				profiler.stop("parseMoves");
				auto errs = validateGame(gamestart, processor.getMoveStr(), moves);
				if (errs.size() > 0) {
					for (auto [gameid, err]: errs) {
						std::cout << err << std::endl;
					}
					throw std::runtime_error("evaluation failed");
				}
				welos->push_back((short)processor.getWelo());
				belos->push_back((short)processor.getBelo());
				gamestarts->push_back(mvids->size());
				mvids->insert(mvids->end(), moves.begin(), moves.end());
				clktimes->insert(clktimes->end(), clk.begin(), clk.end());
				
				int ngames = gamestarts->size();
				int totalGamesEst = ngames / ((float)bytesProcessed / nbytes);
				int curProg = int((100.0f / printFreq) * ngames / totalGamesEst);
				if (curProg > progress) {
					progress = curProg;
					std::string eta = getEta(totalGamesEst, ngames, start);
					std::string status = "parsed " + std::to_string(ngames) + \
										 " games (" + std::to_string(printFreq*progress) + \
										 "% done, eta: " + eta + ")";
					std::cout << status << '\r' << std::flush;
				}
				gamestart = lineno+1;
			}
		}
	}
	return ParserOutput(welos, belos, gamestarts, mvids, clktimes);	
}
