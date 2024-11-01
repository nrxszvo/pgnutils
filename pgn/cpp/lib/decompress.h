#include "zstd.h"
#include <fstream>
#include <ios>
#include <sstream>

void test();

class DecompressStream {	
	ZSTD_DCtx* dctx;
	ZSTD_inBuffer in;
	ZSTD_outBuffer out;
	char* in_mem;
	char* out_mem;
	std::ifstream infile;
	std::stringstream ss;
	size_t zstdRet;
	size_t frameSize;
	std::string rem;
public:
	DecompressStream(std::string zstfn, size_t frameSize=1024*1024);
	~DecompressStream();
	std::streamsize decompressFrame();
	std::vector<std::string> getOutput();
	size_t getFrameSize();
};	
