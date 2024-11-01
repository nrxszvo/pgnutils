#include "decompress.h"
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iostream>

DecompressStream::DecompressStream(std::string zstfn, size_t frameSize) : frameSize(frameSize), rem("") {
	in = {.src = NULL};
    out = {.dst = NULL};

	infile = std::ifstream(zstfn, std::ios::binary);	

 	in_mem = (char*)malloc(frameSize);	
	in.src = in_mem;
	in.size = frameSize;
	in.pos = 0;
	out.dst = malloc(frameSize);
	out.size = frameSize;
	out.pos = 0;

	dctx = ZSTD_createDCtx();
	
	zstdRet = -1;

}

DecompressStream::~DecompressStream() {
	free(in_mem);
	free(out_mem);
	infile.close();
}

std::streamsize DecompressStream::decompressFrame() {
	infile.read(static_cast<char*>(in_mem), frameSize);
	std::streamsize bytesRead = infile.gcount();
	if (bytesRead < 0) throw std::runtime_error("bytesRead < 0");
	in.size = bytesRead;
	in.pos = 0;

	while(true) {
		if (zstdRet == 0 && in.pos == in.size) break;

		out.pos = 0;
		zstdRet = ZSTD_decompressStream(dctx, &out, &in); 
		if (ZSTD_isError(zstdRet)) throw std::runtime_error("zstd returned " + std::to_string(zstdRet));
		ss.write(reinterpret_cast<const char*>(out.dst), out.pos);

		if (in.pos == in.size) {
			if (bytesRead == 0 && zstdRet == 0 && out.pos == out.size) {
				continue;
			}
			break;
		}
	}
	return bytesRead;
}

std::vector<std::string> DecompressStream::getOutput() {
	std::string frame = ss.str();
	ss.str(std::string());

	std::vector<std::string> lines;
	int offset = 0;
	int next = frame.find('\n', offset);
	if (next == std::string::npos) {
		rem += frame;
		return lines;
	}

	rem += frame.substr(offset, next);
	lines.push_back(rem);
	rem = "";
	offset = next+1;

	while(true) {
		next = frame.find('\n', offset);
		if (next == std::string::npos) {
			rem = frame.substr(offset);
		   	break;
		}
		std::string line = frame.substr(offset, next-offset);
		lines.push_back(line);
		offset = next+1;
	}
	return lines;
}

size_t DecompressStream::getFrameSize() {
	return frameSize;
}

void test() {
	DecompressStream decompressor("../../lichess_db_standard_rated_2013-01.pgn.zst");
	while ((decompressor.decompressFrame()) != 0) {
		std::vector<std::string> lines = decompressor.getOutput();
	}
}
