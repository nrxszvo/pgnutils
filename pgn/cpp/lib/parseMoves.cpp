#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <regex>
#include <re2/re2.h>
#include <tuple>
#include "parseMoves.h"
#include "inference.h"

using namespace std;

const string MV_PAT = "O-O-O|O-O|[a-hRNBQK]+[0-9=x]*[a-hRNBQK]*[0-9]*[=RNBQ]*";
const string CLK_PAT = "\\{.*\\[%clk ([0-9:]+)\\].*\\}";
const re2::RE2 twoMoves("\\..* (" + MV_PAT + ").*" + CLK_PAT + ".* (" + MV_PAT + ").*" + CLK_PAT);
const re2::RE2 oneMove("\\..* (" + MV_PAT + ").*" + CLK_PAT);

string movenoToStr(int moveno) {
	return to_string(moveno) + ". ";
}

tuple<int, vector<string>, long> matchNextMove(string& moveStr, int idx, int curmv) {
	int mvstart = idx;
	string nextmv = movenoToStr(curmv+1);
	while(idx < moveStr.size() && moveStr.substr(idx, nextmv.size()) != nextmv) {
		idx++;
	}
	string wm, wclk, bm, bclk;
	string ss = moveStr.substr(mvstart, idx-mvstart);
	auto ps = chrono::high_resolution_clock::now();
	bool found2 = re2::RE2::PartialMatch(ss, twoMoves, &wm, &wclk, &bm, &bclk);
	if (!found2) {
		bool found1 = re2::RE2::PartialMatch(ss, oneMove, &wm, &wclk);
		if (!found1) throw runtime_error("matchNextMove failed");
	}
	auto pe = chrono::high_resolution_clock::now();
	long total_ns = chrono::duration_cast<chrono::nanoseconds>(pe-ps).count();
	vector<string> matches;
	if (found2) {
		matches = {wm, wclk, bm, bclk};
	} else {
		matches = {wm, wclk};
	}
	return make_tuple(idx, matches, total_ns);
}

int clkToSec(string timeStr) {
	int m = stoi(timeStr.substr(2, 2));
	int s = stoi(timeStr.substr(5, 2));
	return m * 60 + s;
}


tuple<vector<int>, vector<int>, int, int, long > parseMoves(string moveStr) {
	vector<int> mvids;
	vector<int> clk;
	int curmv = 1;
	int idx = 0;
	vector<string> matches;
	MoveParser parser;
	int mnm_ns = 0;
	int iid_ns = 0;
	long regex_ns;
	long total_re_ns = 0;
	while (idx < moveStr.size()) {
		auto ps = chrono::high_resolution_clock::now();
		tie(idx, matches, regex_ns) = matchNextMove(moveStr, idx, curmv);
		auto pe = chrono::high_resolution_clock::now();
		mnm_ns += chrono::duration_cast<chrono::nanoseconds>(pe-ps).count();
		total_re_ns += regex_ns;

		if (idx == moveStr.size() && matches.size() == 0) {
			break;
		}
		string wm = matches[0];
		auto is = chrono::high_resolution_clock::now();
		mvids.push_back(parser.inferId(wm));
		auto ie = chrono::high_resolution_clock::now();
		iid_ns += chrono::duration_cast<chrono::nanoseconds>(ie-is).count();

		clk.push_back(clkToSec(matches[1]));
		if (matches.size() == 4) {
			string bm = matches[2];
			mvids.push_back(parser.inferId(bm));
			clk.push_back(clkToSec(matches[3]));
		}
		curmv++;
	}
	return make_tuple(mvids, clk, mnm_ns, iid_ns, total_re_ns);
}

const string TERM_PATS[] = {
	"Normal",
	"Time forfeit"
};

const re2::RE2 timeRe("\\[TimeControl \"([0-9]+)\\+*([0-9]+)\"\\]");
const re2::RE2 termRe("\\[Termination \"(.+)\"\\]");
const re2::RE2 reW("\\[WhiteElo \"([0-9]+)\"\\]");
const re2::RE2 reB("\\[BlackElo \"([0-9]+)\"\\]");

string processRawLine(string& line, State& state) {
	line.erase(remove(line.begin(), line.end(), '\n'), line.cend());
	if (line.size() > 0) {
		if (line[0] == '[') {
			if (line.substr(0, 6) == "[Event") {
				state.init();
			} else if (line.substr(0, 9) == "[WhiteElo") {
				state.weloStr = line;
			} else if (line.substr(0,9) == "[BlackElo") {
				state.beloStr = line;
			} else if (line.substr(0,12) == "[TimeControl") {
				int tim, inc = 0;
				if (re2::RE2::PartialMatch(line, timeRe, &tim, &inc)) {
					if (inc == 0 && tim <= 1200 && tim >= 600) {
						state.time = tim;
					}
				}
			} else if (line.substr(0, 12) == "[Termination") {
				string term;
				re2::RE2::PartialMatch(line, termRe, &term);
				for (auto tp: TERM_PATS) {
					if (term.find(tp) != string::npos) {
						state.validTerm = true;
						break;
					}
				}
			}
		} else if (line[0] == '1') {
			if (state.time > 0 && state.weloStr != "" && state.beloStr != "") {
				int welo, belo;
				bool haveW = re2::RE2::PartialMatch(state.weloStr, reW, &welo);
				bool haveB = re2::RE2::PartialMatch(state.beloStr, reB, &belo);
				if (haveW && haveB) {
					state.welo = welo;
					state.belo = belo;
					state.moveStr = line;
					return "COMPLETE";
				}
			}
			return "INVALID";
		}
	}
	return "INCOMPLETE";
}

PgnProcessor::PgnProcessor(): reinit(false) {}

string PgnProcessor::processLine(string& line) {
	if (this->reinit) {
		this->state.init();
		this->reinit = false;
	}
	string code = processRawLine(line, this->state);
	if (code == "COMPLETE" || code == "INVALID") {
		this->reinit = true;
	}
	return code;
}
int PgnProcessor::getWelo() {
	return this->state.welo;
}
int PgnProcessor::getBelo() {
	return this->state.belo;
}
string PgnProcessor::getMoveStr() {
	return this->state.moveStr;
}
int PgnProcessor::getTime() {
	return this->state.time;
}
