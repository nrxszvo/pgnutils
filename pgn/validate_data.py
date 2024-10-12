import numpy as np
import argparse
import parse_pgn

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
    if mvid in [parse_pgn.QCASTLEW, parse_pgn.QCASTLEB]:
        return "O-O-O"
    elif mvid in [parse_pgn.KCASTLEW, parse_pgn.KCASTLEB]:
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
    try:
        piece, file, rank = pfr
    except Exception as e:
        print(e)
        return False

    if piece == "P":
        return f"{file}{rank}" in mv
    else:
        return piece in mv and file in mv and rank in mv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", help="pgn filename", required=True)
    parser.add_argument("--npy", help="npy name", required=True)

    args = parser.parse_args()

    print("loading inputs...")
    mdfile = f"{args.npy}_md.npy"
    movefile = f"{args.npy}_moves.npy"
    md = np.load(mdfile, allow_pickle=True).item()
    moves = np.memmap(movefile, mode="r", dtype="int32", shape=md["shape"])

    fin = open(args.pgn, "r")
    game = 0
    lineno = 0
    gamestart = 0
    while True:
        state = parse_pgn.init_state()
        for line in fin:
            lineno += 1
            code = parse_pgn.process_raw_line(line, state)
            if code in [parse_pgn.COMPLETE, parse_pgn.INVALID]:
                if code == parse_pgn.COMPLETE:
                    break
                else:
                    gamestart = lineno + 1
                    state = parse_pgn.init_state()
        else:
            break  # EOF

        if code == parse_pgn.COMPLETE:
            move_str = state["move_str"][:-5]
            curmv = 1
            mv_idx = 0

            game_md = md["games"][game]
            start = game_md["start"]
            end = start + game_md["length"]
            mvids = moves[start:end]

            game += 1
            id_idx = 0
            print(f"evaluating game {game}", end="\r")
            while mv_idx < len(move_str):
                mv_idx, m = parse_pgn.match_next_move(move_str, mv_idx, curmv)
                for mv in m.groups():
                    mvid = mvids[id_idx]
                    id_idx += 1
                    pfr = decode_mvid(mvid)
                    if not compare_moves(mv, pfr):
                        print(
                            f"Move mismatch: game {game} ({gamestart}), move {curmv}, {mv} != {pfr}"
                        )
                        return
                curmv += 1

        gamestart = lineno + 1
    print("\nvalidation PASSED")


if __name__ == "__main__":
    main()
