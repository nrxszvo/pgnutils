#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <regex>
#include "parseMoves.h"

using namespace std;

const string MV_PAT = "O-O-O|O-O|[a-hRNBQK]+[0-9=x]*[a-hRNBQK]*[0-9]*[=RNBQ]*";
const string CLK_PAT = "\\{.*\\[%clk ([0-9:]+)\\].*\\}";

string movenoToStr(int moveno) {
	return to_string(moveno) + ". ";
}

pair<int, vector<string>> matchNextMove(string& moveStr, int idx, int curmv) {
	int mvstart = idx;
	string nextmv = movenoToStr(curmv+1);
	while(idx < moveStr.size() && moveStr.substr(idx, nextmv.size()) != nextmv) {
		idx++;
	}
	smatch m;
	string re = to_string(curmv) + "\\..* (" + MV_PAT + ").*" + CLK_PAT + ".* (" + MV_PAT + ").*" + CLK_PAT;
	regex twoMoves(re);
	string ss = moveStr.substr(mvstart, idx-mvstart);
	bool found = regex_search(ss, m, twoMoves);
	if (!found) {
		re = to_string(curmv) + "\\..* (" + MV_PAT + ").*" + CLK_PAT;
		regex oneMove(re);
		found = regex_search(ss, m, oneMove);
		if (!found) throw runtime_error("matchNextMove failed");
	}
	vector<string> matches;
	for (int i=0; i<m.size(); i++) {
		matches.push_back(m[i].str());
	}
	return make_pair(idx, matches);
}

int clk_to_sec(string timeStr) {
	int m = atoi(timeStr.substr(2, 2));
	int s = atoi(timeStr.substr(5, 2));
	return m * 60 + s;
}

const string[] TERM_PATS = [
	"Normal",
	"Time forfeit"
];

struct State {
	string weloStr;
	string beloStr;
	int welo;
	int belo;
	int time;
	bool validTerm;
	string moveStr;
	State(): weloStr(""), belowStr(""), welo(0), belo(0), time(0), validTerm(false), moveStr("") {};
	void init() {
		self.weloStr = "";
		self.beloStr = "";
		self.welo = 0;
		self.belo = 0;
		self.time = 0;
		self.validTerm = false;
		self.moveStr = "";
	};
};

string processRawLine(string& line, State& state) {
	erase(remove(line.begin(), line.end(), '\n'), line.cend());
	if (line.size() > 0) {
		if (line[0] == '[') {
			if (line.substr(0, 6) == "[Event") {
				state.init();
			} else if (line.substr(0, 9) == "[WhiteElo") {
				self.weloStr = line;
			} else if (line.substr(0,9) == "[BlackElo") {
				self.beloStr = line;
			} else if (line.substr(0,12) == "[TimeControl") {
				smatch m;
				regex re("\[TimeControl \"([0-9]+)\+*([0-9]+)\"\]");
				if (regex_search(line, m, re)) {
					int tim = atoi(m[1].str());
					int inc = 0;
					if (m.size() > 2) {
						inc = atoi(m[2].str());
					}
					if (inc == 0 && time <= 1200 && time >= 600) {
						state.time = tim;
					}
				}
			} else if (line.substr(0, 12) == "[Termination") {
				smatch m;
				regex re("\[Termination \"(.+)\"\]");
				regex_search(line, m, re);
				for (auto tp: TERM_PATS) {
					if (m[1].find(tp) != string::npos) {
						state.validTerm = true;
						break;
					}
				}
			}
		} else if (line[0] == '1') {
			if (state.time > 0 && state.weloStr != "" && state.beloStr != "") {
				smatch wm, bm;
				regex reW("\[WhiteElo \"([0-9]+)\"\]");
				bool haveW = regex_search(state.weloStr, wm, reW);
				regex reB("\[BlackElo \"([0-9]+)\"\]");
				bool haveB = regex_search(state.beloStr, bm, reB);
				if (haveW && haveB) {
					state.welo = atoi(wm[1].str());
					state.belo = atoi(bm[1].str());
					state.moveStr = line;
					return "COMPLETE":
				}
			}
			return "INVALID";
		}
	}
	return "INCOMPLETE";
}
