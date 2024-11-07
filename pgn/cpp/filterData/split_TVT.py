import numpy as np
import argparse
import os


def load_npy(indir):
    gs = np.load(f"{indir}/gs.npy", allow_pickle=True)
    ge = np.load(f"{indir}/ge.npy", allow_pickle=True)
    elo = np.load(f"{indir}/elo.npy", allow_pickle=True)
    return gs, ge, elo


def expand_and_split(gs, ge, elo, minMoves, trainp, testp):
    big_n = 0
    for start, end in zip(gs, ge):
        big_n += end + 1 - start - minMoves

    ngames = gs.shape[0]
    ntrain = int(trainp * ngames)
    ntest = int(testp * ngames)
    nval = ngames - ntrain - ntest

    train = np.empty((big_n, 2), dtype="int64")
    val = np.empty((big_n, 2), dtype="int64")
    test = np.empty((big_n, 2), dtype="int64")

    randidx = np.random.choice(range(ngames), size=ngames, replace=False)

    ind = [0, 0, 0]
    for i, r in enumerate(randidx):
        start = gs[r]
        end = ge[r]
        n = end + 1 - start - minMoves
        if i < ntrain:
            arr = train
            idx = 0
        elif i < ntrain + nval:
            arr = val
            idx = 1
        else:
            arr = test
            idx = 2

        arr[ind[idx] : ind[idx] + n, 0] = r
        arr[ind[idx] : ind[idx] + n, 1] = np.arange(minMoves, minMoves + n)

        if i % 1000 == 0:
            print(f"{int(100*i/len(gs))}%", end="\r")
        ind[idx] += n

    train = train[: ind[0]]
    val = val[: ind[1]]
    test = test[: ind[2]]

    return train, val, test


def write_out(outdir, train, val, test):
    for data, name in [(train, "train"), (val, "val"), (test, "test")]:
        np.save(os.path.join(outdir, f"{name}.npy"), data, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", help="input directory")
    parser.add_argument("--outdir", help="output directory")
    parser.add_argument("--trainP", default=0.8, type=float, help="trainP")
    parser.add_argument("--testP", default=0.1, type=float, help="testP")
    parser.add_argument(
        "--minMoves", default=11, type=int, help="minimum number of moves"
    )
    args = parser.parse_args()

    gs, ge, elo = load_npy(args.indir)
    train, val, test = expand_and_split(
        gs, ge, elo, args.minMoves, args.trainP, args.testP
    )
    write_out(args.outdir, train, val, test)


if __name__ == "__main__":
    main()
