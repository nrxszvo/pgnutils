#include <string>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#include "parseMoves.h"
#include "validate.h"

int main() {
	using nano = std::chrono::nanoseconds;
	PgnProcessor processor;
	std::ifstream infile("../../test.pgn");
	std::string line;
	int gamestart = 0;
	int lineno = 0;
	std::vector<int> welos;
	std::vector<int> belos;
	std::vector<long> gamestarts;
	std::vector<std::vector<int> > mvids;
	std::vector<std::vector<int> > clktimes;
	long nmoves = 0;

	float avg_ms = 0;
	float avg_mnm_ms = 0;
	float avg_iid_ms = 0;
	float avg_re_ms = 0;
	auto start = std::chrono::high_resolution_clock::now();
	while (std::getline(infile, line)) {
		lineno++;
		std::string code = processor.processLine(line);		
		if (code == "COMPLETE") {
			auto ps = std::chrono::high_resolution_clock::now();
			auto [moves, clk, mnm_ns, iid_ns, re_ns] = parseMoves(processor.getMoveStr());
			auto pe = std::chrono::high_resolution_clock::now();
			int total_ns = std::chrono::duration_cast<nano>(pe-ps).count();
			avg_ms = 0.9*avg_ms + 0.0000001*total_ns;
			avg_mnm_ms = 0.9*avg_mnm_ms + 0.0000001*mnm_ns;
			avg_iid_ms = 0.9*avg_iid_ms + 0.0000001*iid_ns;
			avg_re_ms = 0.9*avg_re_ms + 0.0000001*re_ns;
			auto errs = validateGame(gamestart, processor.getMoveStr(), moves);
			if (errs.size() > 0) {
				for (auto [gameid, err]: errs) {
					std::cout << err << std::endl;
				}
				throw std::runtime_error("evaluation failed");
			}
			welos.push_back(processor.getWelo());
			belos.push_back(processor.getBelo());
			gamestarts.push_back(nmoves);
			mvids.push_back(moves);
			clktimes.push_back(clk);

			if (gamestarts.size() % 1000 == 0) {
				std::cout << "processed " << std::to_string(gamestarts.size()) << " games\r" << std::flush;
			}
			gamestart = lineno+1;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	long total_ns = std::chrono::duration_cast<nano>(end-start).count();
	int total_s = int(total_ns/1e9);
	int min = total_s/60;
	size_t nzero = 2;
	auto sec = std::to_string(total_s % 60);
	auto seczp = std::string(nzero - std::min(nzero, sec.size()), '0') + sec;
	std::cout << std::endl << min << ":" << seczp << " to process" << std::endl;
	std::cout << "average parseMoves ms: " << avg_ms << std::endl;
	std::cout << "average matchNextMove ms: " << avg_mnm_ms << std::endl;
	std::cout << "average inferId ms: " << avg_iid_ms << std::endl;
	std::cout << "average regex ms: " << avg_re_ms << std::endl;
	return 0;
}
