from . import inference as inf
import re

MV_PAT = "O-O-O|O-O|[a-hRNBQK]+[0-9=x]*[a-hRNBQK]*[0-9]*[=RNBQ]*"


def moveno_str(moveno):
    return f"{moveno}. "


def match_next_move(move_str, idx, curmv):
    mvstart = idx
    nextmv = moveno_str(curmv + 1)
    while idx < len(move_str) and move_str[idx : idx + len(nextmv)] != nextmv:
        idx += 1
    m = re.match(f"{curmv}\..* ({MV_PAT}).* ({MV_PAT})", move_str[mvstart:idx])
    if m is None:
        m = re.match(f"{curmv}\..* ({MV_PAT})", move_str[mvstart:idx])
    return idx, m


def parse_moves(move_str):
    board, white, black = inf.board_state()
    mvids = []
    bm = None
    curmv = 1
    idx = 0

    while idx < len(move_str):
        idx, m = match_next_move(move_str, idx, curmv)
        if idx == len(move_str):
            break
        wm = m.group(1)
        mvdata = inf.parse_move(wm, bm, inf.COLORW)
        mvid = inf.infer_mvid(mvdata, board, white, black)
        mvids.append(mvid)

        if len(m.groups()) == 2:
            bm = m.group(2)
            mvdata = inf.parse_move(bm, wm, inf.COLORB)
            mvid = inf.infer_mvid(mvdata, board, black, white)
            mvids.append(mvid)

        curmv += 1

    return mvids


def init_state(state={}):
    state["welo"] = None
    state["belo"] = None
    state["valid_time"] = False
    state["valid_term"] = False
    state["move_str"] = None
    return state


TERM_PATS = [
    "Normal",
    "Time forfeit",
    "won on time",
    "won by resignation",
    "won by checkmate",
    "Game drawn",
]


def process_raw_line(line, state):
    line = line.strip()
    if len(line) > 0:
        if line[0] == "[":
            if line[:6] == "[Event":
                init_state(state)
            elif line[:9] == "[WhiteElo":
                state["welo"] = line
            elif line[:9] == "[BlackElo":
                state["belo"] = line
            elif line[:12] == "[TimeControl":
                if line in ['[TimeControl "600+0"]', '[TimeControl "600"]']:
                    state["valid_time"] = True
            elif line[:12] == "[Termination":
                m = re.match('\[Termination "(.+)"\]', line)
                for tp in TERM_PATS:
                    if tp in m.group(1):
                        state["valid_term"] = True
                        break
        elif line[0] == "1":
            if state["valid_time"] and state["valid_term"]:
                mw = re.match('\[WhiteElo "([0-9]+)"\]', state["welo"])
                mb = re.match('\[BlackElo "([0-9]+)"\]', state["belo"])
                if mw and mb:
                    state["welo"] = int(mw.group(1))
                    state["belo"] = int(mb.group(1))
                    state["move_str"] = line
                    return "COMPLETE"
            return "INVALID"
    return "INCOMPLETE"


class PgnProcessor:
    def __init__(self):
        self.state = init_state()
        self.reinit = False

    def process_line(self, line):
        if self.reinit:
            init_state(self.state)
            self.reinit = False
        code = process_raw_line(line, self.state)
        if code in ["COMPLETE", "INVALID"]:
            self.reinit = True

        return code

    def get_welo(self):
        return self.state["welo"]

    def get_belo(self):
        return self.state["belo"]

    def get_move_str(self):
        return self.state["move_str"]
