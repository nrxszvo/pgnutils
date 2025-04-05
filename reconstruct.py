from . import inference as inf
from . import validate as val
import chess
import chess.pgn
from collections import Counter


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


def enpassant(piece, board, sr, sf):
    if piece.name != 'P':
        return False
    dr = piece.rank
    df = piece.file

    if piece.color*(dr-sr) != 1:
        return False
    if abs(df-sf) != 1:
        return False

    return board[dr][df] is None and board[sr][df] and board[sr][df].name == 'P' and board[sr][df].color == -piece.color


def update_to_uci(update, board, white, black, update_state=True, promote="q"):
    pid, dr, df = update

    if pid < 16:
        state = white
    else:
        pid -= 16
        state = black

    if state[pid].captured:
        raise CapturedMoveException

    sr = state[pid].rank
    sf = state[pid].file
    if update_state:
        state[pid].rank = dr
        state[pid].file = df
        if board[dr][df] is not None:
            board[dr][df].captured = True
        if enpassant(state[pid], board, sr, sf):
            board[sr][df].captured = True
            board[sr][df] = None

        board[sr][sf] = None
        board[dr][df] = state[pid]

    sr = val.int_to_rank(sr)
    dr = val.int_to_rank(dr)
    sf = val.INT_TO_FILE[sf]
    df = val.INT_TO_FILE[df]

    src = f"{sf}{sr}"
    dst = f"{df}{dr}"

    uci = f"{src}{dst}"

    if is_null(uci):
        raise NullMoveException

    if promotion(state[pid], dr):
        if update_state:
            state[pid].name = promote.upper()
        uci += promote

    return uci


def mvid_to_uci(mvid, board_state, white, black, update_state=True, promote="q"):
    updates = decode_mvid(mvid)
    uci = update_to_uci(updates[0], board_state,
                        white, black, update_state, promote)
    if len(updates) == 2:
        update_to_uci(updates[1], board_state, white, black, update_state)
    return uci


def uci_to_castle_id(uci, white, black):
    if len(uci) == 4:
        sf, sr, df, dr = uci
        if sf == "e":
            if sr == "1" and dr == "1" and white[inf.KING].pos() == (0, 4):
                if df == "g":
                    return inf.KCASTLEW
                elif df == "c":
                    return inf.QCASTLEW
            elif sr == "8" and dr == "8" and black[inf.KING].pos() == (7, 4):
                if df == "g":
                    return inf.KCASTLEB
                elif df == "c":
                    return inf.QCASTLEB
    return None


def uci_to_mvid(uci, white, black):
    mvid = uci_to_castle_id(uci, white, black)
    if mvid is None:
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


class IllegalBoardException(IllegalMoveException):
    pass


class NullMoveException(IllegalMoveException):
    pass


class CapturedMoveException(IllegalMoveException):
    pass


def is_null(uci):
    half = int(len(uci) / 2)
    return uci[:half] == uci[half:]


class BoardState:
    def __init__(self):
        self.state, self.white, self.black = inf.board_state()
        self.game = chess.pgn.Game()
        self.board = self.game.board()

    def uci_to_mvid(self, uci):
        return uci_to_mvid(uci, self.white, self.black)

    def print(self):
        line = "\n"
        for rank in reversed(self.state):
            for piece in rank:
                if piece is None or piece.captured:
                    line += " ."
                else:
                    if piece.color == inf.COLORW:
                        name = piece.name.upper()
                    else:
                        name = piece.name.lower()
                    line += name.rjust(2)
            line += "\n"
        return line

    def mvid_to_uci(self, mvid):
        uci = mvid_to_uci(mvid, self.state, self.white, self.black, False)
        return uci

    def update(self, mvid):
        uci = mvid_to_uci(mvid, self.state, self.white, self.black, False)
        mv = chess.Move.from_uci(uci)
        if self.board.is_legal(mv):
            mvid_to_uci(mvid, self.state, self.white, self.black)
            self.board.push(mv)
        else:
            raise IllegalMoveException(
                f"illegal move {uci} for board:\n{self.board}"
            )
        return mv


exporter = chess.pgn.StringExporter(
    headers=False, variations=False, comments=False)


def pid_to_name(pid):
    if pid >= 16:
        pid -= 16
    if pid in [0, 7]:
        return 'R'
    elif pid in [1, 6]:
        return 'N'
    elif pid in [2, 5]:
        return 'B'
    elif pid == 3:
        return 'Q'
    elif pid == 4:
        return 'K'
    else:
        return 'P'


def try_king_in_check(piece, dr, df, board, board_state, white, black):
    idx = piece.pid if piece.pid < 16 else piece.pid - 16
    cur = white if piece.pid < 16 else black
    opp = white if piece.pid >= 16 else black

    sr = cur[idx].rank
    sf = cur[idx].file

    if cur[idx] != board_state[sr][sf]:
        raise Exception

    cur[idx].rank = dr
    cur[idx].file = df

    board_state[sr][sf] = None
    sav = board_state[dr][df]
    orig = False
    if sav:
        orig = sav.captured
        sav.captured = True
    board_state[dr][df] = cur[idx]

    result = inf.king_in_check(board_state, cur, opp)

    if sav:
        sav.captured = orig
    board_state[dr][df] = sav
    board_state[sr][sf] = cur[idx]

    cur[idx].rank = sr
    cur[idx].file = sf

    return result


def get_invalid_reason(mvid, board, board_state, white, black):
    data = decode_mvid(mvid)
    if len(data) == 2:
        return 'CASTLE'

    pid, dr, df = data[0]
    if pid < 16:
        piece = white[pid]
    else:
        piece = black[pid-16]

    if try_king_in_check(piece, dr, df, board, board_state, white, black):
        return 'CHECK'
    elif piece.name == "P":
        return 'PAWN'
    elif piece.name == "R":
        assert not inf.legal_rook_move(piece, board_state, dr, df)
        return 'ROOK'
    elif piece.name == "N":
        assert not inf.legal_knight_move(piece, board_state, dr, df)
        return 'KNIGHT'
    elif piece.name == "B":
        assert not inf.legal_bishop_move(piece, board_state, dr, df)
        return 'BISHOP'
    elif piece.name == "Q":
        assert not inf.legal_queen_move(piece, board_state, dr, df)
        return 'QUEEN'
    elif piece.name == "K":
        assert not inf.legal_king_move(piece, board_state, dr, df)
        return 'KING'


def print_board_state(board_state):
    for row in reversed(board_state):
        line = ''
        for p in row:
            if p is None:
                line += '. '
            elif p.color == inf.COLORB:
                line += f'{p.name.lower()} '
            else:
                line += f'{p.name} '
        print(line)


def compare_board_to_board(board, board_state):
    fen = board.fen()
    bfen = fen.split(' ')[0]
    rows = bfen.split('/')
    for i, row in enumerate(reversed(rows)):
        srow = board_state[i]
        j = 0
        for c in row:
            try:
                for ii in range(j, j+int(c)):
                    if srow[ii] is not None:
                        raise Exception('board/board_state mismatch')
                    j += 1
            except ValueError:
                if srow[j] is None:
                    raise Exception('board/board_state mismatch')
                if c == c.lower():
                    if srow[j].color != inf.COLORB:
                        raise Exception('board/board_state mismatch')
                else:
                    if srow[j].color != inf.COLORW:
                        raise Exception('board/board_state mismatch')

                if srow[j].name != c.upper():
                    raise Exception('board/board_state mismatch')
                j += 1


def count_invalid(top_n_mvids, opening, tgts):
    board_state, white, black = inf.board_state()
    game = chess.pgn.Game()
    board = game.board()
    reasons = Counter()
    for mvid in opening:
        uci = mvid_to_uci(mvid, board_state, white, black)
        board.push(chess.Move.from_uci(uci))

    top_n, nmoves = top_n_mvids.shape
    nfails = [0]*top_n
    for i in range(nmoves):
        tgt = tgts[i]
        if tgt == inf.NOOP:
            nmoves = i
            break

        for j in range(top_n):
            mvid = top_n_mvids[j, i]
            try:
                uci = mvid_to_uci(mvid, board_state, white, black, False)
                if not board.is_legal(chess.Move.from_uci(uci)):
                    raise IllegalBoardException
                break
            except IllegalMoveException as e:
                if isinstance(e, NullMoveException):
                    reason = 'NULL'
                elif isinstance(e, CapturedMoveException):
                    reason = 'CAPTURED'
                elif isinstance(e, IllegalBoardException):
                    reason = get_invalid_reason(
                        mvid, board, board_state, white, black)
                else:
                    raise e

                reasons[reason] += 1
                nfails[j] += 1

        uci = mvid_to_uci(tgt, board_state, white, black)
        board.push(chess.Move.from_uci(uci))
        compare_board_to_board(board, board_state)

    return nmoves, nfails, reasons


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
