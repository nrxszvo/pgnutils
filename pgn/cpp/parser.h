#ifndef MC_PARSER_H
#define MC_PARSER_H
#include <memory>

struct ParserOutput {
	int64_t ngames;
	int64_t nmoves;
	std::shared_ptr<std::vector<int16_t> > welos;
	std::shared_ptr<std::vector<int16_t> > belos;
	std::shared_ptr<std::vector<int64_t> > gamestarts;
	std::shared_ptr<std::vector<int16_t> > mvids;
	std::shared_ptr<std::vector<int16_t> > clk;
};
#endif
