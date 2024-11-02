#include <string>
#include <iostream>
#include <chrono>
#include "npy.hpp"
#include "parallelParser.h"
#include "serialParser.h"
#include "lib/parseMoves.h"
#include "lib/validate.h"
#include "profiling/profiler.h"
#include "lib/utils.h"
#include "lib/decompress.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, zst, "", ".zst archive to decompress and parse");
ABSL_FLAG(std::string, name, "", "human-readable name for archive");
ABSL_FLAG(std::string, outdir, ".", "output directory to store npy output files");
ABSL_FLAG(bool, serial, false, "Disable parallel processing");

void writeNpy(std::string outdir, ParserOutput& res) {

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
	gs_ptr.shape = { res.gamestarts->size() };
	mv_ptr.data_ptr = moves.data();
	mv_ptr.shape = { 2, res.mvids->size() };

	npy::write_npy(outdir + "/elos.npy", elo_ptr);
	npy::write_npy(outdir + "/gamestarts.npy", gs_ptr);
	npy::write_npy(outdir + "/moves.npy", mv_ptr);
}

int main(int argc, char *argv[]) {
	absl::ParseCommandLine(argc, argv);
	auto start = std::chrono::high_resolution_clock::now();
	ParserOutput res;
	if (absl::GetFlag(FLAGS_serial)) {
		res = processSerial(absl::GetFlag(FLAGS_zst));
	} else {
		ParallelParser parser(std::thread::hardware_concurrency()-1);
		res = parser.parse(absl::GetFlag(FLAGS_zst), absl::GetFlag(FLAGS_name));
	}
	writeNpy(absl::GetFlag(FLAGS_outdir), res);
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << std::endl << "Total processing time: " << getEllapsedStr(start, stop) << std::endl;
	profiler.report();
	return 0;
}
