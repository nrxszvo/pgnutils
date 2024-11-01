#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include "npy.hpp"
#include "parallelParser.h"
#include "parseMoves.h"
#include "validate.h"
#include "profiler.h"
#include "utils.h"
#include "decompress.h"

namespace fs = std::filesystem;

void writeNpy(std::string outdir, Result& res) {

	std::vector<int16_t> elos(res.welos->size() + res.belos->size());
	elos.insert(elos.begin(), res.welos->begin(), res.welos->end());
	elos.insert(elos.begin() + res.welos->size(), res.belos->begin(), res.belos->end());
	
	std::vector<int16_t> moves(2 * res.mvids->size());
	moves.insert(moves.begin(), res.mvids->begin(), res.mvids->end());
	moves.insert(moves.begin() + res.mvids->size(), res.clk->begin(), res.clk->end());
	
	npy::npy_data_ptr<int16_t> elo_ptr;
	npy::npy_data_ptr<int64_t> gs_ptr; 
	npy::npy_data_ptr<int16_t> mv_ptr;
	
	elo_ptr.data_ptr = elos.data(); 	
	elo_ptr.shape = { 2, res.welos->size() };
	gs_ptr.data_ptr = res.gamestarts->data();
	mv_ptr.data_ptr = moves.data();
	mv_ptr.shape = { 2, res.mvids->size() };

	npy::write_npy(outdir + "/elos.npy", elo_ptr);
	npy::write_npy(outdir + "/gamestarts.npy", gs_ptr);
	npy::write_npy(outdir + "/moves.npy", mv_ptr);
}

int main(int argc, char *argv[]) {
	auto start = std::chrono::high_resolution_clock::now();

	ParallelParser parser(std::thread::hardware_concurrency()-1);
	Result res = parser.parse(argv[1], "test");
	writeNpy(argv[2], res);

	auto stop = std::chrono::high_resolution_clock::now();
	int ellapsed = std::chrono::duration_cast<std::chrono::seconds>(stop-start).count();
	int hrs = ellapsed/3600;	
	int minutes = (ellapsed % 3600) / 60;
	int secs = ellapsed % 60;
	std::cout << std::endl << "Total processing time: " << hrs << ":" << zfill(minutes) << ":" << zfill(secs) << std::endl;
	return 0;
}

void processSerial(std::string pgn) {
	PgnProcessor processor;
	uintmax_t nbytes = fs::file_size(pgn);
	std::ifstream infile(pgn);
	std::string line;
	int gamestart = 0;
	int lineno = 0;
	std::vector<int16_t> welos;
	std::vector<int16_t> belos;
	std::vector<int64_t> gamestarts;
	std::vector<int16_t> mvids;
	std::vector<int16_t> clktimes;
	
	int64_t nmoves = 0;
	uintmax_t bytesProcessed = 0;
	profiler.init("parseMoves");
	profiler.init("matchNextMove");
	profiler.init("inferId");
	profiler.init("regex");
	profiler.init("decompress");
	profiler.init("main", 1);
	profiler.start("main");
	auto start = hrc::now();
	while (std::getline(infile, line)) {
		lineno++;
		bytesProcessed += line.size();
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
			welos.push_back((short)processor.getWelo());
			belos.push_back((short)processor.getBelo());
			gamestarts.push_back(nmoves);
			mvids.insert(mvids.end(), moves.begin(), moves.end());
			clktimes.insert(clktimes.end(), clk.begin(), clk.end());

			if (gamestarts.size() % 1000 == 0) {
				std::string eta = getEta(nbytes, bytesProcessed, start);
				std::cout << "processed " << std::to_string(gamestarts.size()) << " games (eta: " << eta << ")\r" << std::flush;
			}
			gamestart = lineno+1;
		}
	}

	profiler.stop("main");
	std::cout << std::endl;
	profiler.report();

	std::vector<int16_t> elos(welos.size() + belos.size());
	elos.insert(elos.begin(), welos.begin(), welos.end());
	elos.insert(elos.begin() + welos.size(), belos.begin(), belos.end());
	
	std::vector<int16_t> moves(2 * mvids.size());
	moves.insert(moves.begin(), mvids.begin(), mvids.end());
	moves.insert(moves.begin() + mvids.size(), clktimes.begin(), clktimes.end());
	
	npy::npy_data_ptr<int16_t> elo_ptr;
	npy::npy_data_ptr<int64_t> gs_ptr; 
	npy::npy_data_ptr<int16_t> mv_ptr;
	
	elo_ptr.data_ptr = elos.data(); 	
	elo_ptr.shape = { 2, welos.size() };
	gs_ptr.data_ptr = gamestarts.data();
	mv_ptr.data_ptr = moves.data();
	mv_ptr.shape = { 2, mvids.size() };

	npy::write_npy("elos.npy", elo_ptr);
	npy::write_npy("gamestarts.npy", gs_ptr);
	npy::write_npy("moves.npy", mv_ptr);
}
