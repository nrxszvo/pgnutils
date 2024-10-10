import argparse
import re


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


def SQR_TO_INT(sqr):
    return FILE_TO_INT[sqr[0]] + 8 * (int(sqr[1]) - 1)


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
                ret["src_file"] = mv[0]
                ret["dest"] = mv[2:]
                if (
                    len(last_mv) == 2
                    and last_mv[0] == mv[2]
                    and int(last_mv[1]) + color == int(mv[3])
                ):
                    ret["enpassant"] = True
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
                ret["src_file"] = mv[1]
            ret["dest"] = mv[2:]
        elif len(mv) == 5:
            if mv[2] == "x":
                ret["src_file"] = mv[1]
                ret["dest"] = mv[3:5]
            else:
                ret["src"] = mv[1:3]
                ret["dest"] = mv[3:5]
        else:
            assert len(mv) == 6, f"unexpected mv length: {mv}"
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
            return [4 * 64 + 6, 7 * 64 + 5]
        else:
            board[7][4] = None
            board[7][6] = state[4]
            board[7][7] = None
            board[7][5] = state[7]
            return [4 * 64 + 63, 7 * 64 + 62]
    else:
        state[4].file = 2
        state[0].file = 3
        if state[0].color == 1:
            board[0][4] = None
            board[0][2] = state[4]
            board[0][0] = None
            board[0][3] = state[0]
            return [4 * 64 + 2, 3]
        else:
            board[7][4] = None
            board[7][2] = state[4]
            board[7][0] = None
            board[7][3] = state[0]
            return [4 * 64 + 58, 59]


def legal_pawn_move(piece, board, dr, df, enpassant):
    sr = piece.rank
    sf = piece.file
    if sf == df:
        if abs(dr - sr) > 2:
            return False
        if abs(dr - sr) == 2:
            if piece.color == 1 and dr != 3:
                return False
            if piece.color == -1 and dr != 4:
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
    sr = piece.rank
    sf = piece.file
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
    sr = piece.rank
    sf = piece.file

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
    sr = piece.rank
    sf = piece.file

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
    dr = int(dest[1]) - 1
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
    dr = b.rank
    df = b.file

    if a.name == "P":
        sr = a.rank
        sf = a.file
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


def infer_piece(mvdata, board, state, opp_state):
    if "castle" in mvdata:
        return parse_castle(mvdata, board, state)

    candidates = []
    if "src" in mvdata:
        for i in range(len(state)):
            piece = state[i]
            sr = int(mvdata["src"][1]) - 1
            sf = FILE_TO_INT[mvdata["src"][0]]
            if piece.rank == sr and piece.file == sf:
                if "promotion" in mvdata:
                    piece.name = mvdata["piece"]
                candidates.append(i)

    elif "src_file" in mvdata:
        for i in range(len(state)):
            piece = state[i]
            if (
                board[piece.rank][piece.file] == piece
                and piece.name == mvdata["piece"]
                and FILE_TO_INT[mvdata["src_file"]] == piece.file
                and legal_move(piece, board, mvdata["dest"], "enpassant" in mvdata)
            ):
                candidates.append(i)

    else:
        for i in range(len(state)):
            piece = state[i]
            if (
                board[piece.rank][piece.file] == piece and piece.name == mvdata["piece"]
            ) and legal_move(piece, board, mvdata["dest"]):
                candidates.append(i)

    dr = int(mvdata["dest"][1]) - 1
    df = FILE_TO_INT[mvdata["dest"][0]]

    valid = []
    for idx in candidates:
        sr = state[idx].rank
        sf = state[idx].file
        state[idx].rank = dr
        state[idx].file = df
        board[sr][sf] = None
        temp = board[dr][df]
        if temp:
            temp.captured = True
        board[dr][df] = state[idx]
        if king_in_check(board, state, opp_state):
            board[sr][sf] = state[idx]
            if temp:
                temp.captured = False
            board[dr][df] = temp
            state[idx].rank = sr
            state[idx].file = sf
        else:
            valid.append(idx)

    if len(valid) != 1:
        raise Exception("could not resolve multiple possible moves")

    idx = valid[0]
    state[idx].rank = dr
    state[idx].file = df

    mvid = idx * 64 + SQR_TO_INT(mvdata["dest"])
    return [mvid]


def parse_moves(move_str):
    board, white, black = board_state()
    moves = move_str.split(" ")
    moves = moves[:-1]
    out = []
    bm = None
    for i in range(0, len(moves), 3):
        wm = moves[i + 1]
        mvdata = parse_move(wm, bm, 1)
        mvid = infer_piece(mvdata, board, white, black)
        out.append(mvid)
        if len(moves) > i + 2:
            bm = moves[i + 2]
            mvdata = parse_move(bm, wm, -1)
            mvids = infer_piece(mvdata, board, black, white)
            out.extend(mvids)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", help="pgn filename", required=True)
    parser.add_argument("--csv", help="csv output filename", required=True)

    args = parser.parse_args()

    with open(args.pgn) as fin:
        data = fin.readlines()

    WELO = 7
    BELO = 8
    TC = 13
    TERM = 14
    MVS = 16
    BS = 18

    npdata = []

    for i in range(0, len(data), BS):
        if data[i + TERM] != '[Termination "Normal"]\n':
            continue
        if data[i + TC] != '[TimeControl "600+0"]\n':
            continue

        we = re.match('\[WhiteElo "([0-9]+)"\]', data[i + WELO]).group(1)
        be = re.match('\[BlackElo "([0-9]+)"\]', data[i + BELO]).group(1)

        npdata.append(
            {
                "WhiteElo": int(we),
                "BlackElo": int(be),
                "Moves": parse_moves(data[i + MVS]),
            }
        )


if __name__ == "__main__":
    main()
