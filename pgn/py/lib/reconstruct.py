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


def update_to_uci(update, white, black, update_state=True):
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

    sr = val.int_to_rank(sr)
    dr = val.int_to_rank(dr)
    sf = val.INT_TO_FILE[sf]
    df = val.INT_TO_FILE[df]

    src = f"{sf}{sr}"
    dst = f"{df}{dr}"

    return f"{src}{dst}"


def mvid_to_uci(mvid, white, black, update_state=True):
    updates = decode_mvid(mvid)
    uci = update_to_uci(updates[0], white, black, update_state)
    if len(updates) == 2:
        update_to_uci(updates[1], white, black, update_state)
    return uci


def count_invalid(mvids, opening, tgts):
    _, white, black = inf.board_state()
    game = chess.pgn.Game()
    node = game
    nfail = 0
    for mvid in opening:
        uci = mvid_to_uci(mvid, white, black)
        node = node.add_variation(chess.Move.from_uci(uci))

    nmoves = len(mvids)
    for i, (mvid, tgt) in enumerate(zip(mvids, tgts)):
        if tgt == inf.NOOP:
            nmoves = i
            break
        try:
            uci = mvid_to_uci(mvid, white, black, False)
            if not node.board().is_legal(chess.Move.from_uci(uci)):
                nfail += 1
        except chess.InvalidMoveError:
            nfail += 1

        uci = mvid_to_uci(tgt, white, black)
        node = node.add_variation(chess.Move.from_uci(uci))

    return nmoves, nfail


def reconstruct(mvids):
    _, white, black = inf.board_state()
    game = chess.pgn.Game()
    node = game
    uci_str = []
    success = True
    err = None
    nvalid = len(mvids)
    try:
        for i, mvid in enumerate(mvids):
            uci = mvid_to_uci(mvid, white, black)
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
