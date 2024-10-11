import argparse
import re
import numpy as np
import os
import time
import datetime


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
        board[1][f] = Piece("P", 1, f, f, 1)
        board[6][f] = Piece("P", 6, f, f, -1)

    board[0][QROOK] = Piece("R", 0, 0, 0, 1)
    board[0][QKNIGHT] = Piece("N", 0, 1, 1, 1)
    board[0][QBISHOP] = Piece("B", 0, 2, 2, 1)
    board[0][QUEEN] = Piece("Q", 0, 3, 3, 1)
    board[0][KING] = Piece("K", 0, 4, 4, 1)
    board[0][KBISHOP] = Piece("B", 0, 5, 5, 1)
    board[0][KKNIGHT] = Piece("N", 0, 6, 6, 1)
    board[0][KROOK] = Piece("R", 0, 7, 7, 1)
    board[7][QROOK] = Piece("R", 7, 0, 8, -1)
    board[7][QKNIGHT] = Piece("N", 7, 1, 9, -1)
    board[7][QBISHOP] = Piece("B", 7, 2, 10, -1)
    board[7][QUEEN] = Piece("Q", 7, 3, 11, -1)
    board[7][KING] = Piece("K", 7, 4, 12, -1)
    board[7][KBISHOP] = Piece("B", 7, 5, 13, -1)
    board[7][KKNIGHT] = Piece("N", 7, 6, 14, -1)
    board[7][KROOK] = Piece("R", 7, 7, 15, -1)

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


def parse_move(mv, last_mv, color):
    ret = {}

    if mv[-1] in ["#", "+"]:
        mv = mv[:-1]

    if mv == "O-O":
        ret["castle"] = "king"
        return ret
    elif mv == "O-O-O":
        ret["castle"] = "queen"
        return ret

    if mv[0] >= "a" and mv[0] <= "h":
        ret["piece"] = "P"
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
    else:
        ret["piece"] = mv[0]
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


def parse_castle(mvdata, board, state):
    if mvdata["castle"] == "king":
        state[4].file = 6
        state[7].file = 5

        if state[0].color == 1:
            board[0][4] = None
            board[0][6] = state[4]
            board[0][7] = None
            board[0][5] = state[7]
            return QBISHOP * 64 + 7
        else:
            board[7][4] = None
            board[7][6] = state[4]
            board[7][7] = None
            board[7][5] = state[7]
            return QBISHOP * 64 + 63
    else:
        state[4].file = 2
        state[0].file = 3
        if state[0].color == 1:
            board[0][4] = None
            board[0][2] = state[4]
            board[0][0] = None
            board[0][3] = state[0]
            return KBISHOP * 64
        else:
            board[7][4] = None
            board[7][2] = state[4]
            board[7][0] = None
            board[7][3] = state[0]
            return KBISHOP * 64 + 56


def legal_pawn_move(piece, board, dr, df, enpassant):
    sr, sf = piece.pos()
    if sf == df:
        if abs(dr - sr) > 2:
            return False
        if abs(dr - sr) == 2:
            if piece.color == 1 and (sr != 1 or dr != 3):
                return False
            if piece.color == -1 and (sr != 6 or dr != 4):
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


def infer_piece(mvdata, board, state, opp_state):
    if "castle" in mvdata:
        return parse_castle(mvdata, board, state)

    candidates = []

    for i in range(len(state)):
        piece = state[i]
        if piece.captured:
            continue

        if "src" in mvdata:
            sr, sf = mv_to_pos(mvdata["src"])
            if piece.rank == sr and piece.file == sf:
                if "promotion" in mvdata:
                    if piece.name == "P":
                        piece.name = mvdata["piece"]
                        candidates.append(i)
                else:
                    candidates.append(i)

        elif "src_file" in mvdata or "src_rank" in mvdata:
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
                        candidates.append(i)
                else:
                    candidates.append(i)

        else:
            if (
                board[piece.rank][piece.file] == piece and piece.name == mvdata["piece"]
            ) and legal_move(piece, board, mvdata["dest"]):
                candidates.append(i)

    dr, df = mv_to_pos(mvdata["dest"])

    def update_state(idx, enpassant):
        sr, sf = state[idx].pos()
        state[idx].rank = dr
        state[idx].file = df
        board[sr][sf] = None
        if enpassant:
            tr = dr - state[idx].color
            temp = board[tr][df]
            board[tr][df] = None
        else:
            temp = board[dr][df]
        if temp:
            temp.captured = True
        board[dr][df] = state[idx]
        return sr, sf, temp

    def revert_state(idx, sr, sf, temp, enpassant):
        board[sr][sf] = state[idx]
        if temp:
            temp.captured = False
            board[temp.rank][temp.file] = temp
        if temp is None or enpassant:
            board[dr][df] = None
        state[idx].rank = sr
        state[idx].file = sf

    valid = []
    ep = "enpassant" in mvdata
    for idx in candidates:
        sr, sf, temp = update_state(idx, ep)
        if not king_in_check(board, state, opp_state):
            valid.append(idx)
        revert_state(idx, sr, sf, temp, ep)

    if len(valid) != 1:
        raise Exception("could not resolve possible moves")

    idx = valid[0]
    update_state(idx, ep)
    mvid = idx * 64 + sqr_to_int(mvdata["dest"])
    return mvid


MV_PAT = "O-O-O|O-O|[a-hRNBQK]+[0-9=x]*[a-hRNBQK]*[0-9]*[=RNBQ]*"


def parse_moves(move_str):
    def moveno_str(moveno):
        return f"{moveno}. "

    board, white, black = board_state()

    move_str = move_str[:-5]
    out = []
    bm = None
    curmv = 1
    idx = 0

    while idx < len(move_str) - 3:
        mvstart = idx
        nextmv = moveno_str(curmv + 1)
        while idx < len(move_str) and move_str[idx : idx + len(nextmv)] != nextmv:
            idx += 1
        m = re.match(f"{curmv}\..* ({MV_PAT}).* ({MV_PAT})", move_str[mvstart:idx])
        if m is None:
            m = re.match(f"{curmv}\..* ({MV_PAT})", move_str[mvstart:idx])
        wm = m.group(1)
        mvdata = parse_move(wm, bm, 1)
        mvid = infer_piece(mvdata, board, white, black)
        out.append(mvid)

        if len(m.groups()) == 2:
            bm = m.group(2)
            mvdata = parse_move(bm, wm, -1)
            mvid = infer_piece(mvdata, board, black, white)
            out.append(mvid)

        curmv += 1

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", help="pgn filename", required=True)
    parser.add_argument("--out", help="npy output filename", required=True)

    args = parser.parse_args()

    npdata = []

    game = 0

    # for debugging
    lineno = 0
    gamestart = 0

    nbytes = os.path.getsize(args.pgn)
    bytes_processed = 0

    fin = open(args.pgn, "r")
    start = time.time()
    eta = "tbd"
    while True:
        move_str = None
        welo = None
        belo = None
        validTime = False
        validTerm = False

        # for debugging
        data = []

        for line in fin:
            bytes_processed += len(line)
            data.append(line)
            lineno += 1
            if line[0] == "[":
                if line[:9] == "[WhiteElo":
                    welo = line
                elif line[:9] == "[BlackElo":
                    belo = line
                elif line[:12] == "[TimeControl":
                    if line == '[TimeControl "600+0"]\n':
                        validTime = True
                elif line[:12] == "[Termination":
                    m = re.match('\[Termination "(.+)"\]\n', line)
                    if m.group(1) in ["Normal", "Time forfeit"]:
                        validTerm = True
            elif line[0] == "1":
                gamestart = lineno + 1
                if validTime and validTerm:
                    mw = re.match('\[WhiteElo "([0-9]+)"\]', welo)
                    mb = re.match('\[BlackElo "([0-9]+)"\]', belo)
                    if mw and mb:
                        we = mw.group(1)
                        be = mb.group(1)
                        move_str = line
                        break

                validTime = False
                validTerm = False
                data = []
        else:
            break  # EOF

        if move_str:
            npdata.append(
                {
                    "WhiteElo": int(we),
                    "BlackElo": int(be),
                    "Moves": parse_moves(move_str),
                }
            )

            game += 1
            if game % 100 == 0:
                end = time.time()
                eta = datetime.timedelta(
                    seconds=(nbytes - bytes_processed) * (end - start) / bytes_processed
                )
                hours = eta.seconds // 3600
                minutes = (eta.seconds % 3600) // 60
                seconds = eta.seconds % 60
                eta_str = f"{eta.days}:{hours}:{minutes:02}:{seconds:02}"
                print(
                    f"processed {game} games (eta: {eta_str})",
                    end="\r",
                )

    fin.close()
    print(f"Number of games: {len(npdata)}")
    np.save(args.out, npdata, allow_pickle=True)


if __name__ == "__main__":
    main()
