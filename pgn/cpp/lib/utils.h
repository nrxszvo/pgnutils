#include <chrono>
#include <string>

using hrc = std::chrono::high_resolution_clock;
using tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using nano = std::chrono::nanoseconds;

std::string getEta(uintmax_t total, uintmax_t soFar, tp &start);
