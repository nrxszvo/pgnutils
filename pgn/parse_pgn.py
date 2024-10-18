import argparse
import re
import numpy as np
import os
import time
import datetime
from multiprocessing import Process, Queue, Lock
import traceback


def sign(v):
    return 1 if v > 0 else -1 if v < 0 else 0


class Piece:
    def __init__(self, name, rank, file, pid, color):
        self.name = name
        self.rank = rank
        self.file = file
        self.pid = pid
        self.color = color
        self.captured = False

    def pos(self):
        return (self.rank, self.file)


COLORW = 1
COLORB = -1
QROOK = 0
QKNIGHT = 1
QBISHOP = 2
QUEEN = 3
KING = 4
KBISHOP = 5
KKNIGHT = 6
KROOK = 7


def board_state():
    board = [[None for _ in range(8)] for _ in range(8)]
    white = []
    black = []

    for f in range(8):
        board[1][f] = Piece("P", 1, f, 8 + f, COLORW)
        board[6][f] = Piece("P", 6, f, 24 + f, COLORB)

    board[0][QROOK] = Piece("R", 0, 0, 0, COLORW)
    board[0][QKNIGHT] = Piece("N", 0, 1, 1, COLORW)
    board[0][QBISHOP] = Piece("B", 0, 2, 2, COLORW)
    board[0][QUEEN] = Piece("Q", 0, 3, 3, COLORW)
    board[0][KING] = Piece("K", 0, 4, 4, COLORW)
    board[0][KBISHOP] = Piece("B", 0, 5, 5, COLORW)
    board[0][KKNIGHT] = Piece("N", 0, 6, 6, COLORW)
    board[0][KROOK] = Piece("R", 0, 7, 7, COLORW)
    board[7][QROOK] = Piece("R", 7, 0, 16, COLORB)
    board[7][QKNIGHT] = Piece("N", 7, 1, 17, COLORB)
    board[7][QBISHOP] = Piece("B", 7, 2, 18, COLORB)
    board[7][QUEEN] = Piece("Q", 7, 3, 19, COLORB)
    board[7][KING] = Piece("K", 7, 4, 20, COLORB)
    board[7][KBISHOP] = Piece("B", 7, 5, 21, COLORB)
    board[7][KKNIGHT] = Piece("N", 7, 6, 22, COLORB)
    board[7][KROOK] = Piece("R", 7, 7, 23, COLORB)

    for i in range(2):
        for j in range(8):
            white.append(board[i][j])
            black.append(board[7 - i][j])

    return board, white, black


FILE_TO_INT = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
}


def rank_to_int(r):
    return int(r) - 1


def sqr_to_int(sqr):
    return FILE_TO_INT[sqr[0]] + 8 * rank_to_int(sqr[1])


def parse_rank_or_file(rorf, ret):
    if rorf >= "a" and rorf <= "h":
        ret["src_file"] = rorf
    else:
        ret["src_rank"] = rorf


def parse_pawn_move(mv, last_mv, color):
    ret = {"piece": "P"}
    if len(mv) == 2:
        ret["dest"] = mv
    else:
        if mv[1] == "x":
            parse_rank_or_file(mv[0], ret)
            ret["dest"] = mv[2:4]
            if (
                len(last_mv) == 2
                and last_mv[0] == mv[2]
                and int(last_mv[1]) + color == int(mv[3])
            ):
                ret["enpassant"] = True
            if len(mv) > 4 and mv[4] == "=":
                ret["promotion"] = True
                ret["piece"] = mv[5]
        elif mv[2] == "=":
            ret["dest"] = mv[:2]
            ret["src"] = mv[0] + str(int(mv[1]) - color)
            ret["piece"] = mv[3]
            ret["promotion"] = True
        elif mv[1] >= "1" and mv[1] <= "8":
            ret["src"] = mv[:2]
            ret["dest"] = mv[3:5]
        else:
            raise Exception(f"pawn parse error: {mv}")
    return ret


def parse_non_pawn_move(mv):
    ret = {"piece": mv[0]}
    if len(mv) == 3:
        ret["dest"] = mv[1:]
    elif len(mv) == 4:
        if mv[1] != "x":
            parse_rank_or_file(mv[1], ret)
        ret["dest"] = mv[2:]
    elif len(mv) == 5:
        if mv[2] == "x":
            parse_rank_or_file(mv[1], ret)
            ret["dest"] = mv[3:5]
        else:
            ret["src"] = mv[1:3]
            ret["dest"] = mv[3:5]
    else:
        if len(mv) != 6:
            raise Exception(f"unexpected mv length: {mv}")
        ret["src"] = mv[1:3]
        ret["dest"] = mv[4:6]
    return ret


def parse_move(mv, last_mv, color):
    if mv == "O-O":
        return {"castle": "king"}
    elif mv == "O-O-O":
        return {"castle": "queen"}
    elif mv[0] >= "a" and mv[0] <= "h":
        return parse_pawn_move(mv, last_mv, color)
    else:
        return parse_non_pawn_move(mv)


KCASTLEW = QBISHOP * 64 + 7
KCASTLEB = (16 + QBISHOP) * 64 + 63
QCASTLEW = KBISHOP * 64
QCASTLEB = (16 + KBISHOP) * 64 + 56


def castle_to_mvid(mvdata, board, state):
    if mvdata["castle"] == "king":
        state[KING].file = 6
        state[KROOK].file = 5

        if state[0].color == COLORW:
            board[0][4] = None
            board[0][6] = state[KING]
            board[0][7] = None
            board[0][5] = state[KROOK]
            return KCASTLEW
        else:
            board[7][4] = None
            board[7][6] = state[KING]
            board[7][7] = None
            board[7][5] = state[KROOK]
            return KCASTLEB
    else:
        state[KING].file = 2
        state[QROOK].file = 3
        if state[0].color == COLORW:
            board[0][4] = None
            board[0][2] = state[KING]
            board[0][0] = None
            board[0][3] = state[QROOK]
            return QCASTLEW
        else:
            board[7][4] = None
            board[7][2] = state[KING]
            board[7][0] = None
            board[7][3] = state[QROOK]
            return QCASTLEB


def legal_pawn_move(piece, board, dr, df, enpassant):
    sr, sf = piece.pos()
    if sf == df:
        if abs(dr - sr) > 2:
            return False
        if abs(dr - sr) == 2:
            if piece.color == COLORW and (sr != 1 or dr != 3):
                return False
            if piece.color == COLORB and (sr != 6 or dr != 4):
                return False
            if board[dr - piece.color][df] is not None:
                return False
        elif piece.color * (dr - sr) != 1:
            return False
        return board[dr][df] is None
    elif abs(df - sf) == 1:
        if piece.color * (dr - sr) != 1:
            return False
        if enpassant:
            return board[sr][df] is not None and board[sr][df].color == -piece.color
        else:
            return board[dr][df] is not None and board[dr][df].color == -piece.color
    else:
        return False


def legal_rook_move(piece, board, dr, df):
    sr, sf = piece.pos()
    if sf != df and sr != dr:
        return False

    if sf != df:
        for i in range(sf + sign(df - sf), df, sign(df - sf)):
            if board[dr][i] is not None:
                return False
    else:
        for i in range(sr + sign(dr - sr), dr, sign(dr - sr)):
            if board[i][df] is not None:
                return False

    return board[dr][df] is None or board[dr][df].color != piece.color


def legal_knight_move(piece, board, dr, df):
    sr, sf = piece.pos()

    if abs(sf - df) == 2:
        if abs(sr - dr) != 1:
            return False
    elif abs(sr - dr) == 2:
        if abs(sf - df) != 1:
            return False
    else:
        return False

    return board[dr][df] is None or board[dr][df].color != piece.color


def legal_bishop_move(piece, board, dr, df):
    sr, sf = piece.pos()

    if abs(sf - df) != abs(sr - dr):
        return False

    finc = sign(df - sf)
    rinc = sign(dr - sr)
    for i in range(1, abs(sf - df)):
        r = sr + rinc * i
        f = sf + finc * i
        if board[r][f] is not None:
            return False

    return board[dr][df] is None or board[dr][df].color != piece.color


def legal_queen_move(piece, board, dr, df):
    return legal_rook_move(piece, board, dr, df) or legal_bishop_move(
        piece, board, dr, df
    )


def legal_move(piece, board, dest, enpassant=False):
    dr = rank_to_int(dest[1])
    df = FILE_TO_INT[dest[0]]

    if piece.name == "P":
        return legal_pawn_move(piece, board, dr, df, enpassant)
    elif piece.name == "R":
        return legal_rook_move(piece, board, dr, df)
    elif piece.name == "N":
        return legal_knight_move(piece, board, dr, df)
    elif piece.name == "B":
        return legal_bishop_move(piece, board, dr, df)
    elif piece.name == "Q":
        return legal_queen_move(piece, board, dr, df)
    elif piece.name == "K":
        return True


def attacking(a, b, board):
    dr, df = b.pos()
    if a.name == "P":
        sr, sf = a.pos()
        return a.color * (dr - sr) == 1 and abs(df - sf) == 1
    elif a.name == "R":
        return legal_rook_move(a, board, dr, df)
    elif a.name == "N":
        return legal_knight_move(a, board, dr, df)
    elif a.name == "B":
        return legal_bishop_move(a, board, dr, df)
    elif a.name == "Q":
        return legal_queen_move(a, board, dr, df)
    return False


def king_in_check(board, cur, opp):
    king = cur[KING]
    for piece in opp:
        if not piece.captured and attacking(piece, king, board):
            return True
    return False


def mv_to_pos(mv):
    r = int(mv[1]) - 1
    f = FILE_TO_INT[mv[0]]
    return (r, f)


def src_inference_match(piece, mvdata):
    sr, sf = mv_to_pos(mvdata["src"])
    if piece.rank == sr and piece.file == sf:
        if "promotion" in mvdata:
            if piece.name == "P":
                piece.name = mvdata["piece"]
                return True
        else:
            return True
    return False


def src_rf_inference_match(piece, mvdata, board):
    if "src_file" in mvdata:
        src_cond = FILE_TO_INT[mvdata["src_file"]] == piece.file
    else:
        src_cond = rank_to_int(mvdata["src_rank"]) == piece.rank

    if (
        board[piece.rank][piece.file] == piece
        and (
            piece.name == mvdata["piece"]
            or ("promotion" in mvdata and piece.name == "P")
        )
        and src_cond
        and legal_move(piece, board, mvdata["dest"], "enpassant" in mvdata)
    ):
        if "promotion" in mvdata:
            if piece.name == "P":
                piece.name = mvdata["piece"]
                return True
        else:
            return True
    return False


def generic_inference_match(piece, mvdata, board):
    return (
        board[piece.rank][piece.file] == piece
        and piece.name == mvdata["piece"]
        and legal_move(piece, board, mvdata["dest"])
    )


def infer_mvid(mvdata, board, state, opp_state):
    if "castle" in mvdata:
        return castle_to_mvid(mvdata, board, state)

    candidates = []

    def src_im(piece):
        return src_inference_match(piece, mvdata)

    def src_rf_im(piece):
        return src_rf_inference_match(piece, mvdata, board)

    def gen_im(piece):
        return generic_inference_match(piece, mvdata, board)

    if "src" in mvdata:
        cond = src_im
    elif "src_file" in mvdata or "src_rank" in mvdata:
        cond = src_rf_im
    else:
        cond = gen_im

    for piece in state:
        if not piece.captured and cond(piece):
            candidates.append(piece)

    dr, df = mv_to_pos(mvdata["dest"])

    def update_state(piece, enpassant):
        sr, sf = piece.pos()
        piece.rank = dr
        piece.file = df
        board[sr][sf] = None
        if enpassant:
            tr = dr - piece.color
            temp = board[tr][df]
            board[tr][df] = None
        else:
            temp = board[dr][df]
        if temp:
            temp.captured = True
        board[dr][df] = piece
        return sr, sf, temp

    def revert_state(piece, sr, sf, temp, enpassant):
        board[sr][sf] = piece
        if temp:
            temp.captured = False
            board[temp.rank][temp.file] = temp
        if temp is None or enpassant:
            board[dr][df] = None
        piece.rank = sr
        piece.file = sf

    ep = "enpassant" in mvdata
    if len(candidates) > 1:
        valid = []
        for piece in candidates:
            sr, sf, temp = update_state(piece, ep)
            if not king_in_check(board, state, opp_state):
                valid.append(piece)
            revert_state(piece, sr, sf, temp, ep)

        if len(valid) != 1:
            raise Exception("could not resolve possible moves")

        piece = valid[0]

    else:
        piece = candidates[0]

    update_state(piece, ep)
    mvid = piece.pid * 64 + sqr_to_int(mvdata["dest"])
    return mvid


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
    board, white, black = board_state()
    mvids = []
    bm = None
    curmv = 1
    idx = 0

    while idx < len(move_str):
        idx, m = match_next_move(move_str, idx, curmv)
        if idx == len(move_str):
            break
        wm = m.group(1)
        mvdata = parse_move(wm, bm, COLORW)
        mvid = infer_mvid(mvdata, board, white, black)
        mvids.append(mvid)

        if len(m.groups()) == 2:
            bm = m.group(2)
            mvdata = parse_move(bm, wm, COLORB)
            mvid = infer_mvid(mvdata, board, black, white)
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
    "Norma",
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


def get_eta(max_items, items_so_far, start):
    end = time.time()
    eta = datetime.timedelta(
        seconds=(max_items - items_so_far) * (end - start) / items_so_far
    )
    hours = eta.seconds // 3600
    minutes = (eta.seconds % 3600) // 60
    seconds = eta.seconds % 60
    eta_str = f"{eta.days}:{hours}:{minutes:02}:{seconds:02}"
    return eta_str


def load_games(pgn_q, games_q, num_readers, lock):
    while True:
        pgn = pgn_q.get()
        if pgn != "DONE":
            nbytes = os.path.getsize(pgn)
            bytes_processed = 0

            gameid = 0
            with open(pgn) as fin:
                start = time.time()
                while True:
                    state = init_state()
                    for line in fin:
                        bytes_processed += len(line)
                        code = process_raw_line(line, state)
                        if code == "COMPLETE":
                            break
                        elif code == "INVALID":
                            init_state(state)
                    else:
                        break  # EOF

                    if code == "COMPLETE":
                        md = {
                            "WhiteElo": state["welo"],
                            "BlackElo": state["belo"],
                            "gameid": gameid,
                        }
                        gameid += 1
                        games_q.put(("GAME", (md, state["move_str"])))
                        if gameid % 1000 == 0:
                            eta_str = get_eta(nbytes, bytes_processed, start)
                            status_str = (
                                f"parsed {gameid} games (eta: {eta_str})        "
                            )
                            lock.acquire()
                            try:
                                print(status_str, end="\r")
                            finally:
                                lock.release()

        if pgn == "DONE":
            for _ in range(num_readers):
                games_q.put(("SESSION_DONE", None))
            break
        else:
            for _ in range(num_readers):
                games_q.put(("FILE_DONE", None))


def start_games_reader(pgn_q, games_q, n_proc, output_q):
    games_p = Process(target=load_games, args=(pgn_q, games_q, n_proc, output_q))
    games_p.daemon = True
    games_p.start()
    return games_p


PIECE_TO_NAME = [
    "R",
    "N",
    "B",
    "Q",
    "K",
    "B",
    "N",
    "R",
    "P",
    "P",
    "P",
    "P",
    "P",
    "P",
    "P",
    "P",
    "R",
    "N",
    "B",
    "Q",
    "K",
    "B",
    "N",
    "R",
    "P",
    "P",
    "P",
    "P",
    "P",
    "P",
    "P",
    "P",
]

INT_TO_FILE = ["a", "b", "c", "d", "e", "f", "g", "h"]


def int_to_row(r):
    return f"{r+1}"


def decode_mvid(mvid):
    if mvid in [QCASTLEW, QCASTLEB]:
        return "O-O-O"
    elif mvid in [KCASTLEW, KCASTLEB]:
        return "O-O"
    else:
        piece = PIECE_TO_NAME[mvid // 64]
        sqr = mvid % 64
        r = sqr // 8
        f = sqr % 8
        mv = f"{INT_TO_FILE[f]}{int_to_row(r)}"
        return f"{piece}{mv}"


def compare_moves(mv, pfr):
    if mv == pfr:
        return True

    piece, file, rank = pfr

    if piece == "P":
        return f"{file}{rank}" in mv
    else:
        return piece in mv and file in mv and rank in mv


def evaluate_game(gameid, move_str, mvids):
    move_str = move_str[:-5]
    curmv = 1
    mv_idx = 0
    id_idx = 0
    results = []
    while mv_idx < len(move_str):
        try:
            mv_idx, m = match_next_move(move_str, mv_idx, curmv)
            if mv_idx == len(move_str):
                break
            for mv in m.groups():
                mvid = mvids[id_idx]
                id_idx += 1
                pfr = decode_mvid(mvid)
                if not compare_moves(mv, pfr):
                    err = f"Move mismatch: game {gameid}, move {curmv}, {mv} != {pfr}"
                    results.append((gameid, err))
            curmv += 1
        except Exception as e:
            err = str(e)
            results.append((gameid, err))

    return results


def process_games(games_q, output_q, pid, lock):
    while True:
        code, data = games_q.get()
        if code == "FILE_DONE":
            output_q.put(("DONE", None))
        elif code == "SESSION_DONE":
            break

        else:
            md, move_str = data
            mvids = parse_moves(move_str)
            errs = evaluate_game(md["gameid"], move_str, mvids)
            if len(errs) == 0:
                output_q.put(("GAME", (pid, md, mvids)))
            else:
                output_q.put(("ERROR", errs))


def start_reader_procs(num_readers, games_q, output_q, lock):
    procs = []
    for pid in range(num_readers):
        reader_p = Process(target=process_games, args=(games_q, output_q, pid, lock))
        reader_p.daemon = True
        reader_p.start()
        procs.append(reader_p)
    return procs


class ParallelParser:
    def __init__(self, n_proc):
        self.n_proc = n_proc
        self.print_lock = Lock()
        self.pgn_q = Queue()
        self.games_q = Queue()
        self.output_q = Queue()
        self.reader_ps = start_reader_procs(
            n_proc, self.games_q, self.output_q, self.print_lock
        )
        self.game_p = start_games_reader(
            self.pgn_q, self.games_q, n_proc, self.print_lock
        )

    def close(self):
        self.pgn_q.put("DONE")
        self.games_q.close()
        self.output_q.close()
        self.pgn_q.close()
        for rp in self.reader_ps:
            rp.join()
            rp.close()
        self.game_p.join()
        self.game_p.close()

    def parse(self, pgn):
        start = time.time()
        self.pgn_q.put(pgn)
        n_finished = 0
        games = []
        all_mvids = []
        pid_counts = [0] * self.n_proc
        while True:
            code, data = self.output_q.get()
            if code == "DONE":
                n_finished += 1
                if n_finished == self.n_proc:
                    break
            elif code == "ERROR":
                self.print_lock.acquire()
                try:
                    print()
                    for err in data:
                        print(err)
                finally:
                    self.print_lock.release()
            elif code == "GAME":
                pid, md, mvids = data
                pid_counts[pid] += 1
                nmoves = len(all_mvids)
                md["start"] = nmoves
                md["end"] = nmoves + len(mvids)
                games.append(md)
                all_mvids.extend(mvids)
                # self.print_lock.acquire()
                # try:
                #    print(f"pid counts: {pid_counts}", end="\r")
                # finally:
                #    self.print_lock.release()
            else:
                raise Exception(f"invalid code: {code}")

        end = time.time()
        print(
            f"Total time to process pgn: {(end-start)/60:.2f} minutes                       "
        )
        print(f"pid counts: {pid_counts}")
        all_md = {"shape": len(all_mvids), "games": games}
        return all_md, all_mvids


def main_serial(pgn_file=None, pgn_str=None):
    # for debugging
    lineno = 0
    gamestart = 0

    # info
    game = 0
    bytes_processed = 0

    all_moves = []
    md = {"games": []}
    if pgn_file is not None:
        nbytes = os.path.getsize(pgn_file)
        fin = open(pgn_file, "r")

    else:
        nbytes = len(pgn_str)
        fin = pgn_str.split("\n")

    start = time.time()
    nmoves = 0
    while True:
        state = init_state()
        data = []
        for i, line in enumerate(fin):
            bytes_processed += len(line)
            data.append(line)
            lineno += 1
            try:
                code = process_raw_line(line, state)
            except Exception as e:
                print(traceback.format_exc())
                print(e)
            if code in ["COMPLETE", "INVALID"]:
                if code == "COMPLETE":
                    break
                else:
                    state = init_state()
                    data = []
        else:
            break  # EOF

        if pgn_str is not None:
            fin = fin[i:]

        if code == "COMPLETE":
            try:
                mvids = parse_moves(state["move_str"])
                errs = evaluate_game(gamestart, state["move_str"], mvids)
                if len(errs) > 0:
                    for err in errs:
                        print(err)
                    raise Exception("evaluation failed")

                md["games"].append(
                    {
                        "WhiteElo": state["welo"],
                        "BlackElo": state["belo"],
                        "start": nmoves,
                        "length": len(mvids),
                    }
                )
                nmoves += len(mvids)
                all_moves.extend(mvids)
                game += 1
                if game % 100 == 0:
                    eta_str = get_eta(nbytes, bytes_processed, start)
                    print(f"processed {game} games (eta: {eta_str})", end="\r")

            except Exception as e:
                print(e)
                print(f"game start: {gamestart}")

            gamestart = lineno + 1

    if pgn_file is not None:
        fin.close()

    return md, all_moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", help="pgn filename", required=True)
    parser.add_argument("--npy", help="npy output name", required=True)
    parser.add_argument(
        "--n_proc",
        help="number of reader processes",
        default=os.cpu_count() - 1,
        type=int,
    )
    parser.add_argument(
        "--serial", help="run in single process", action="store_true", default=False
    )

    args = parser.parse_args()

    all_md, all_moves = main_serial(pgn_file=args.pgn)

    print(f"\nNumber of games: {len(all_md['games'])}")
    if len(all_md["games"]) > 0:
        mdfile = f"{args.npy}_md.npy"
        mvfile = f"{args.npy}_moves.npy"
        all_md["shape"] = len(all_moves)
        np.save(mdfile, all_md, allow_pickle=True)
        output = np.memmap(mvfile, dtype="int32", mode="w+", shape=all_md["shape"])
        output[:] = all_moves[:]


if __name__ == "__main__":
    main()
