#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "npy.hpp"
#include "parseMoves.h"
#include "validate.h"

#include "profiler.h"

/*
std::string getEta(int total, int soFar, std::chrono::time_point<std::chrono::high_resolution_clock> &start) {	
	if (soFar == 0) {
		return "tbd";
	}
	auto stop = std::chrono::high_resolution_clock::now();
	long ellapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
	long remaining = (total-soFar) * ellapsed / soFar;
	int hrs = remaining / 1e9 / 3600;
	int minutes = ((remaining / 1e9) % 3600) / 60;
	int secs = (remaining / 1e9) % 60;
	return std::to_string(hrs) + "h" + std::to_string(minutes) + "m" + std::to_string(secs);
}
*/

int main() {
	PgnProcessor processor;
	std::ifstream infile("../../test.pgn");
	std::string line;
	int gamestart = 0;
	int lineno = 0;
	std::vector<int16_t> welos;
	std::vector<int16_t> belos;
	std::vector<int64_t> gamestarts;
	std::vector<int16_t> mvids;
	std::vector<int16_t> clktimes;
	
	int64_t nmoves = 0;
	profiler.init("parseMoves");
	profiler.init("matchNextMove");
	profiler.init("inferId");
	profiler.init("regex");
	profiler.init("main", 1);
	profiler.start("main");

	while (std::getline(infile, line)) {
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
			welos.push_back((short)processor.getWelo());
			belos.push_back((short)processor.getBelo());
			gamestarts.push_back(nmoves);
			mvids.insert(mvids.end(), moves.begin(), moves.end());
			clktimes.insert(clktimes.end(), clk.begin(), clk.end());

			if (gamestarts.size() % 1000 == 0) {
				std::cout << "processed " << std::to_string(gamestarts.size()) << " games\r" << std::flush;
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

	return 0;
}
