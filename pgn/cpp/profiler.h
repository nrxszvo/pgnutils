#include <string>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <string>

struct Block {
	float avg;
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	long total_nano;
	int reportFmt;
	Block(): avg(0.0f), total_nano(0), reportFmt(0) {};
	Block(int reportFmt): avg(0.0f), total_nano(0), reportFmt(reportFmt) {};
};

#ifdef PROFILE_ENABLE
class Profiler {
	float alpha;
	std::unordered_map<std::string,Block> blocks;
	std::vector<std::string> names;
public:
	Profiler(float alpha=0.9): alpha(alpha) {};
	void init(std::string name, int reportFmt=0) {
		this->names.push_back(name);
		this->blocks[name] = Block(reportFmt);	
	}
	inline void start(std::string name) {
		this->blocks[name].start = std::chrono::high_resolution_clock::now();	
	}
	inline void stop(std::string name) {
		auto stop = std::chrono::high_resolution_clock::now();
		auto start = this->blocks[name].start;
		long nano = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
		this->blocks[name].total_nano += nano;
	}	
	long getNano(std::string name) {
		return this->blocks[name].total_nano;
	}

	inline void average(std::string name) {
		Block& block = this->blocks[name];
		block.avg = this->alpha*block.avg + (1-this->alpha)*block.total_nano/1e6;	
		block.total_nano = 0;
	}
	float getAverage(std::string name) {
		return this->blocks[name].avg;
	}
	void report() {
		for (auto& name: this->names) {
			Block& block = this->blocks[name];
			std::string val;
			if (block.reportFmt == 0) {
				val = std::to_string(block.avg) + " ms";
			} else {
				float total_s = block.total_nano/1e9;
				int min = int(total_s/60);
				size_t nzero = 2;
				auto sec = total_s - 60*min;
				auto secfmt = std::format("{:.2f}", sec); 
				auto seczp = std::string(nzero - std::min(nzero, std::to_string(int(sec)).size()), '0') + secfmt;
				val = std::to_string(min) + ":" + seczp + " total processing time";
			}
			std::cout << name << ": " << val << std::endl;
		}
	}
};
#else
class Profiler {
public:
	void init(std::string name, int reportFmt=0) {
		return;
	}
	inline void start(std::string name) {
		return;
	}
	inline void stop(std::string name) {
		return;
	}
	inline void average(std::string name) {
		return;
	}
	long getNano(std::string name) {
		return 0;
	}
	float getAverage(std::string name) {
		return 0.0f;
	}
	void report() {
		return;
	}
};
#endif

extern Profiler profiler;
