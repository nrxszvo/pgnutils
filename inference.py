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

NOOP = QBISHOP * 64 + 3
STARTMV = KBISHOP * 64 + 4


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


def legal_king_move(piece, board, dr, df):
    return abs(piece.rank-dr) in [0, 1] and abs(piece.file-df) in [0, 1] and (board[dr][df] is None or board[dr][df].color != piece.color)


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
        return legal_king_move(piece, board, dr, df)


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
    elif a.name == 'K':
        return legal_king_move(a, board, dr, df)
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
