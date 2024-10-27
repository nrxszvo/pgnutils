from . import inference as inf
from .parse_moves import match_next_move

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
    if mvid in [inf.QCASTLEW, inf.QCASTLEB]:
        return "O-O-O"
    elif mvid in [inf.KCASTLEW, inf.KCASTLEB]:
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


def validate_game(gameid, move_str, mvids):
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
            for mv in m.groups()[::2]:
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
            break

    return results
