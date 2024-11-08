import numpy as np
import argparse
import os


def load_npy(indir):
    gs = np.load(f"{indir}/gs.npy", allow_pickle=True)
    ge = np.load(f"{indir}/ge.npy", allow_pickle=True)
    return gs, ge


def expand_and_split(gs, ge, min_mvs, trainp, testp):
    ngames = gs.shape[0]
    ntrain = int(trainp * ngames)
    ntest = int(testp * ngames)
    nval = ngames - ntrain - ntest

    train = np.empty((ntrain, 2), dtype="int64")
    val = np.empty((nval, 2), dtype="int64")
    test = np.empty((ntest, 2), dtype="int64")
    nsamp = [0, 0, 0]
    ridx = np.random.choice(np.arange(ngames), size=ngames, replace=False)
    didx = [0, 0, 0]

    for i, r in enumerate(ridx):
        start = gs[r]
        end = ge[r]
        if i < ntrain:
            ds_idx = 0
            arr = train
        elif i < ntrain + nval:
            ds_idx = 1
            arr = val
        else:
            ds_idx = 2
            arr = test

        idx = didx[ds_idx]
        arr[idx, 0] = nsamp[ds_idx]
        arr[idx, 1] = r
        n = end + 1 - start - min_mvs
        nsamp[ds_idx] += n
        didx[ds_idx] += 1

        if i % 1000 == 0:
            print(f"{int(100*i/len(gs))}%", end="\r")

    return nsamp, train, val, test


def write_out(outdir, ngames, min_mvs, train, train_n, val, val_n, test, test_n):
    np.save(
        f"{outdir}/filter_md.npy",
        {
            "min_moves": min_mvs,
            "train_n": train_n,
            "train_shape": train.shape,
            "val_n": val_n,
            "val_shape": val.shape,
            "test_n": test_n,
            "test_shape": test.shape,
            "ngames": ngames,
        },
        allow_pickle=True,
    )
    for data, name in [(train, "train"), (val, "val"), (test, "test")]:
        mmap = np.memmap(
            os.path.join(outdir, f"{name}.npy"),
            mode="w+",
            dtype="int64",
            shape=data.shape,
        )
        mmap[:] = data[:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", help="input directory")
    parser.add_argument("--outdir", help="output directory")
    parser.add_argument("--trainP", default=0.8, type=float, help="trainP")
    parser.add_argument("--testP", default=0.1, type=float, help="testP")
    parser.add_argument(
        "--min_moves", default=11, type=int, help="minimum number of moves"
    )
    args = parser.parse_args()

    gs, ge = load_npy(args.indir)
    big_n, train, val, test = expand_and_split(
        gs, ge, args.min_moves, args.trainP, args.testP
    )
    write_out(
        args.outdir,
        gs.shape[0],
        args.min_moves,
        train,
        big_n[0],
        val,
        big_n[1],
        test,
        big_n[2],
    )


if __name__ == "__main__":
    main()
