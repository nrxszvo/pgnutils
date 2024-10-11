import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", help="pgn filename", required=True)
    parser.add_argument("--line", help="starting line of game", type=int, required=True)
    parser.add_argument("--out", help="pgn output filename", required=True)

    args = parser.parse_args()

    with open(args.pgn, "r") as fin:
        lineno = 0
        for line in fin:
            lineno += 1
            if lineno == args.line:
                break
        data = []
        for line in fin:
            data.append(line)
            if line[0] == "1":
                break

    with open(args.out, "w") as fout:
        fout.writelines(data)


if __name__ == "__main__":
    main()
