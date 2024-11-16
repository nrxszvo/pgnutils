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


def mvid_to_uci(mvid, white, black):
    updates = decode_mvid(mvid)
    pid, dr, df = updates[0]
    if pid < 16:
        sr = white[pid].rank
        sf = white[pid].file
        white[pid].rank = dr
        white[pid].file = df
    else:
        sr = black[pid - 16].rank
        sf = black[pid - 16].file
        black[pid - 16].rank = dr
        black[pid - 16].file = df

    sr = val.int_to_rank(sr)
    dr = val.int_to_rank(dr)
    sf = val.INT_TO_FILE[sf]
    df = val.INT_TO_FILE[df]

    src = f"{sf}{sr}"
    dst = f"{df}{dr}"

    if len(updates) == 2:
        pid, dr, df = updates[1]
        if pid < 16:
            white[pid].rank = dr
            white[pid].file = df
        else:
            black[pid - 16].rank = dr
            black[pid - 16].file = df

    return f"{src}{dst}"


def reconstruct(mvids):
    _, white, black = inf.board_state()
    game = chess.pgn.Game()
    node = game
    for mvid in mvids:
        uci = mvid_to_uci(mvid, white, black)
        node = node.add_variation(chess.Move.from_uci(uci))
    return str(game.mainline())
