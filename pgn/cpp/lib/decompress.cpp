#include <cstdlib>
#include <stdexcept>
#include <string>
#include <filesystem>
#include "decompress.h"
#include "profiling/profiler.h"

#define MAGIC 0xFD2FB528

std::vector<size_t> getFrameBoundaries(std::string zst, int nBoundaries) {		
	std::vector<size_t> offsets;
	size_t nbytes = std::filesystem::file_size(zst);		
	std::ifstream infile(zst, std::ios::binary);
	int buffer;
	for (int i=0; i<nBoundaries; i++) {
		size_t offset = i*nbytes/nBoundaries;
		infile.seekg(offset, infile.beg);
		while(true) {
			infile.read((char*)(&buffer), 4);
			if (infile.gcount() <= 0) break;
			else if (buffer == MAGIC) {
				offsets.push_back(offset);
				break;
			}
			offset += 4;
		}	
	}
	infile.close();
	offsets.push_back(nbytes);
	return offsets;
}

DecompressStream::DecompressStream(std::string zstfn, size_t frameStart, size_t frameEnd, size_t frameSize) : frameSize(frameSize), rem("") {
	maxBytes = (frameEnd-frameStart);
	totalRead = 0;
	in = {.src = NULL};
    out = {.dst = NULL};

	infile = std::ifstream(zstfn, std::ios::binary);	
	infile.seekg(frameStart, infile.beg);

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
	if (totalRead >= maxBytes) return 0;

	profiler.start("decompress");
	size_t bytesToRead = std::min(frameSize, maxBytes-totalRead);
	infile.read(static_cast<char*>(in_mem), bytesToRead);
	std::streamsize bytesRead = infile.gcount();
	totalRead += bytesRead;
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
	profiler.stop("decompress");
	return bytesRead;
}

std::vector<std::string> DecompressStream::getLines() {
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

float DecompressStream::getProgress() {
	return (float)totalRead / (float)maxBytes;
}

