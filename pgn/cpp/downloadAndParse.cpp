#include <string>
#include <iostream>
#include <vector>
#include "parseMoves.h"

int main() {
	std::string test = "1. e4 { [%clk 0:01:00] } 1... c5 { [%clk 0:01:00] } 2. Nf3 { [%clk 0:00:59] }";
	auto [idx, matches] = matchNextMove(test, 0, 1);
	for (auto m: matches) {
		std::cout << m << std::endl;
	}
	return 0;
}
