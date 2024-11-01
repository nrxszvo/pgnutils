#include "utils.h"

std::string zfill(int time) {
	std::string timeStr = std::to_string(time);
	if (timeStr.size() == 1) timeStr = "0" + timeStr;
	return timeStr;
}


std::string getEta(uintmax_t total, uintmax_t soFar, tp &start) {	
	if (soFar == 0) {
		return "tbd";
	}
	auto stop = hrc::now();
	long ellapsed = std::chrono::duration_cast<milli>(stop-start).count();
	long remaining_ms = (total-soFar) * ellapsed / soFar;
	int remaining = int(remaining_ms/1e3);
	int hrs = remaining / 3600;
	int minutes = (remaining % 3600) / 60;
	int secs = remaining % 60;
	return std::to_string(hrs) + ":" + zfill(minutes) + ":" + zfill(secs);
}
