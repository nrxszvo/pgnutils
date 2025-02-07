from . import inference as inf
from . import validate as val
import chess
import chess.pgn


def decode_mvid(mvid):
    if mvid == inf.QCASTLEW:
        return [(inf.KING, 0, 2), (inf.QROOK, 0, 3)]
    elif mvid == inf.QCASTLEB:
        return [(16 + inf.KING, 7, 2), (16 + inf.QROOK, 7, 3)]
    elif mvid == inf.KCASTLEW:
        return [(inf.KING, 0, 6), (inf.KROOK, 0, 5)]
    elif mvid == inf.KCASTLEB:
        return [(16 + inf.KING, 7, 6), (16 + inf.KROOK, 7, 5)]
    else:
        pid = mvid // 64
        sqr = mvid % 64
        r = sqr // 8
        f = sqr % 8
        return [(pid, r, f)]


def promotion(state, dr):
    if state.name == "P":
        if state.color == inf.COLORW:
            return dr == "8"
        else:
            return dr == "1"


def update_to_uci(update, board, white, black, update_state=True, promote="q"):
    pid, dr, df = update

    if pid < 16:
        state = white
    else:
        pid -= 16
        state = black

    sr = state[pid].rank
    sf = state[pid].file
    if update_state:
        state[pid].rank = dr
        state[pid].file = df
        if board[dr][df] is not None:
            board[dr][df].captured = True
        board[sr][sf] = None
        board[dr][df] = state[pid]

    sr = val.int_to_rank(sr)
    dr = val.int_to_rank(dr)
    sf = val.INT_TO_FILE[sf]
    df = val.INT_TO_FILE[df]

    src = f"{sf}{sr}"
    dst = f"{df}{dr}"

    uci = f"{src}{dst}"

    if promotion(state[pid], dr):
        if update_state:
            state[pid].name = promote.upper()
        uci += promote

    return uci


def mvid_to_uci(mvid, board, white, black, update_state=True, promote="q"):
    updates = decode_mvid(mvid)
    uci = update_to_uci(updates[0], board, white, black, update_state, promote)
    if len(updates) == 2:
        update_to_uci(updates[1], board, white, black, update_state)
    return uci


def uci_to_mvid(uci, white, black):
    sf = inf.FILE_TO_INT[uci[0]]
    sr = inf.rank_to_int(uci[1])
    dst = uci[2:4]
    piece = None
    for p in white + black:
        if not p.captured and p.rank == sr and p.file == sf:
            piece = p
            break
    mvid = piece.pid * 64 + inf.sqr_to_int(dst)
    return mvid


class IllegalMoveException(Exception):
    pass


def is_null(uci):
    half = int(len(uci) / 2)
    return uci[:half] == uci[half:]


class BoardState:
    def __init__(self):
        self.state, self.white, self.black = inf.board_state()
        self.board = chess.pgn.Game().board()

    def uci_to_mvid(self, uci):
        return uci_to_mvid(uci, self.white, self.black)

    def print(self):
        line = "\n"
        for rank in reversed(self.state):
            for piece in rank:
                if piece is None or piece.captured:
                    line += "  "
                else:
                    line += piece.name.rjust(2)
            line += "\n"
        return line

    def update(self, mvid):
        uci = mvid_to_uci(mvid, self.state, self.white, self.black, False)
        if is_null(uci):
            raise IllegalMoveException(f"{uci} is null")
        else:
            mv = chess.Move.from_uci(uci)
            if self.board.is_legal(mv):
                mvid_to_uci(mvid, self.state, self.white, self.black)
                self.board.push(mv)
            else:
                raise IllegalMoveException(
                    f"illegal move {uci} for board:\n{self.board}"
                )
            return mv


def count_invalid(mvids, opening, tgts):
    board_state, white, black = inf.board_state()
    board = chess.pgn.Game().board()
    nfail = 0
    for mvid in opening:
        uci = mvid_to_uci(mvid, board_state, white, black)
        board.push(chess.Move.from_uci(uci))

    nmoves = len(mvids)
    for i, (mvid, tgt) in enumerate(zip(mvids, tgts)):
        if tgt == inf.NOOP:
            nmoves = i
            break
        try:
            uci = mvid_to_uci(mvid, board_state, white, black, False)
            if not board.is_legal(chess.Move.from_uci(uci)):
                nfail += 1
        except chess.InvalidMoveError:
            nfail += 1

        uci = mvid_to_uci(tgt, board_state, white, black)
        try:
            board.push(chess.Move.from_uci(uci))
        except:
            breakpoint()

    return nmoves, nfail


def reconstruct(mvids):
    board, white, black = inf.board_state()
    game = chess.pgn.Game()
    node = game
    uci_str = []
    success = True
    err = None
    nvalid = len(mvids)
    try:
        for i, mvid in enumerate(mvids):
            uci = mvid_to_uci(mvid, board, white, black)
            uci_str.append(uci)
            node = node.add_variation(chess.Move.from_uci(uci))
    except Exception as e:
        success = False
        err = str(e)
        nvalid = i

    return {
        "success": success,
        "err": err,
        "uci": " ".join(uci_str),
        "pgn": str(game.mainline()),
        "nvalidmoves": nvalid,
    }
