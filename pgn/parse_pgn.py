import argparse
import re


def board_state(color):
    ps = 2 if color == 0 else 7
    ms = 1 if color == 0 else 8

    return [
        f"a{ps}",
        f"b{ps}",
        f"c{ps}",
        f"d{ps}",
        f"e{ps}",
        f"f{ps}",
        f"g{ps}",
        f"h{ps}",
        f"a{ms}",
        f"b{ms}",
        f"c{ms}",
        f"d{ms}",
        f"e{ms}",
        f"f{ms}",
        f"g{ms}",
        f"h{ms}",
    ]


PAWN_KEYS = ["a", "b", "c", "d", "e", "f", "g", "h"]


def parse_pawn(mv, state, color):
    sign = -1 if color == 0 else 1
    fil1 = mv[0]
    if mv[1] >= "1" and mv[1] <= "8":
        if len(mv) > 2 and mv[2] == "x":
            src = mv[:2]
            tgt = mv[3:5]
        else:
            fil = mv[0]
            rnk = f"{int(mv[1]) - 1}"
            src = None
            for key in PAWN_KEYS:
                if state[key] == fil + rnk:
                    src = fil + rnk
                    break
            if src is None:
                src = fil + f"{int(rnk) - 1}"
            tgt = mv[:2]
    elif mv[1] == "x":
        src = fil1 + mv[3] + sign
        tgt = mv[2:4]

    for fil in PAWN_KEYS:
        if state[fil] == src:
            state[fil] = tgt
            return (fil, tgt)
    else:
        raise Exception(f"invalid pawn capture: {mv}")


def parse_other(mv, state, color):
    sign = -1 if color == 0 else 1


KEY_TO_INT = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "Ra": 8,
    "Nb": 9,
    "Bc": 10,
    "Q": 11,
    "K": 12,
    "Bf": 13,
    "Ng": 14,
    "Rh": 15,
}


def SQR_TO_INT(sqr):
    return KEY_TO_INT[sqr[0]] + 8 * (int(sqr[1]) - 1)


def parse_moves(move_str, white_state, black_state):
    moves = move_str.split(" ")
    moves = moves[:-1]
    out = []
    for i in range(0, len(moves), 3):
        wm = moves[i + 1]
        if wm[0] >= "a" and wm[0] <= "z":
            key, mv = parse_pawn(wm, white_state, 0)
        else:
            pass
        mvid = KEY_TO_INT[key] * 64 + SQR_TO_INT(mv)
        out.append(mvid)
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

    white_state = board_state(0)
    black_state = board_state(1)

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
                "Moves": parse_moves(data[i + MVS], white_state, black_state),
            }
        )

    print(npdata)


if __name__ == "__main__":
    main()
