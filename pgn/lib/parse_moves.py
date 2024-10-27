from . import inference as inf
import re

MV_PAT = "O-O-O|O-O|[a-hRNBQK]+[0-9=x]*[a-hRNBQK]*[0-9]*[=RNBQ]*"
CLK_PAT = "\{.*\[%clk ([0-9:]+)\].*\}"


def moveno_str(moveno):
    return f"{moveno}. "


def match_next_move(move_str, idx, curmv):
    mvstart = idx
    nextmv = moveno_str(curmv + 1)
    while idx < len(move_str) and move_str[idx : idx + len(nextmv)] != nextmv:
        idx += 1
    m = re.match(
        f"{curmv}\..* ({MV_PAT}).*{CLK_PAT}.* ({MV_PAT}).*{CLK_PAT}",
        move_str[mvstart:idx],
    )
    if m is None:
        m = re.match(f"{curmv}\..* ({MV_PAT}).*{CLK_PAT}", move_str[mvstart:idx])

    return idx, m


def clk_to_sec(time_str):
    m = int(time_str[2:4])
    s = int(time_str[5:7])
    return m * 60 + s


def parse_moves(move_str):
    board, white, black = inf.board_state()
    mvids = []
    clk = []
    bm = None
    curmv = 1
    idx = 0

    while idx < len(move_str):
        idx, m = match_next_move(move_str, idx, curmv)
        if idx == len(move_str) and m is None:
            break
        if len(m.groups()) not in [2, 4]:
            raise Exception("clock field missing")

        wm = m.group(1)
        mvdata = inf.parse_move(wm, bm, inf.COLORW)
        mvid = inf.infer_mvid(mvdata, board, white, black)
        mvids.append(mvid)
        clk.append(clk_to_sec(m.group(2)))

        if len(m.groups()) == 4:
            bm = m.group(3)
            mvdata = inf.parse_move(bm, wm, inf.COLORB)
            mvid = inf.infer_mvid(mvdata, board, black, white)
            mvids.append(mvid)
            clk.append(clk_to_sec(m.group(4)))
        curmv += 1

    return mvids, clk


def init_state(state={}):
    state["welo"] = None
    state["belo"] = None
    state["time"] = 0
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
                m = re.match('\[TimeControl "([0-9]+)\+*([0-9]+)"\]', line)
                if m is not None:
                    tim = int(m.group(1))
                    inc = 0 if len(m.groups()) == 1 else int(m.group(2))
                    if inc == 0 and tim <= 1200 and tim >= 600:
                        state["time"] = tim
            elif line[:12] == "[Termination":
                m = re.match('\[Termination "(.+)"\]', line)
                for tp in TERM_PATS:
                    if tp in m.group(1):
                        state["valid_term"] = True
                        break
        elif line[0] == "1":
            if (
                state["time"] > 0
                and state["valid_term"]
                and state["welo"] is not None
                and state["belo"] is not None
            ):
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

    def get_time(self):
        return self.state["time"]
