#include "utils.h"

std::string getEta(uintmax_t total, uintmax_t soFar, tp &start) {	
	if (soFar == 0) {
		return "tbd";
	}
	auto stop = hrc::now();
	long ellapsed = std::chrono::duration_cast<nano>(stop-start).count();
	long remaining_ns = (total-soFar) * ellapsed / soFar;
	int remaining = int(remaining_ns/1e9);
	int hrs = remaining / 3600;
	int minutes = (remaining % 3600) / 60;
	int secs = remaining % 60;
	return std::to_string(hrs) + "h" + std::to_string(minutes) + "m" + std::to_string(secs);
}
