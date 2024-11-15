import argparse
import numpy as np
import json
import os


def load_data(npydir):
    with open(f"{npydir}/fmd.json") as f:
        fmd = json.load(f)
    elos = np.memmap(
        os.path.join(npydir, "elo.npy"),
        mode="r",
        dtype="int16",
        shape=(fmd["ngames"], 2),
    )
    train = np.memmap(
        os.path.join(npydir, "train.npy"),
        mode="r",
        dtype="int64",
        shape=tuple(fmd["train_shape"]),
    )
    val = np.memmap(
        os.path.join(npydir, "val.npy"),
        mode="r",
        dtype="int64",
        shape=tuple(fmd["val_shape"]),
    )
    test = np.memmap(
        os.path.join(npydir, "test.npy"),
        mode="r",
        dtype="int64",
        shape=tuple(fmd["test_shape"]),
    )
    return fmd, elos, train, val, test


def split_elo(data, elos, elo_lo, elo_hi):
    welos = elos[data[:, 2]][:, 0]
    belos = elos[data[:, 2]][:, 1]

    wfltr = (elo_lo <= welos) & (welos < elo_hi)
    bfltr = (elo_lo <= belos) & (belos < elo_hi)

    codes = np.empty(data.shape[0], dtype=data.dtype)
    codes[wfltr & ~bfltr] = 0
    codes[bfltr & ~wfltr] = 1
    codes[wfltr & bfltr] = 2

    elo_data = data[wfltr | bfltr]
    elo_data[:, 2] = codes[wfltr | bfltr]

    return elo_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npydir")
    parser.add_argument("--elo_edges", nargs="+", type=int)
    parser.add_argument("--outdir")

    args = parser.parse_args()
    npydir = os.path.abspath(args.npydir)
    outdir = os.path.abspath(args.outdir)

    fmd, elos, train, val, test = load_data(npydir)
    elo_lo_hi = [0] + args.elo_edges + [float("inf")]
    for i in range(len(elo_lo_hi) - 1):
        lo = elo_lo_hi[i]
        hi = elo_lo_hi[i + 1]
        tag = f"{hi}" if hi != float("inf") else f"gt-{lo}"
        print(tag)
        tagdir = os.path.join(outdir, tag)
        os.makedirs(tagdir, exist_ok=True)

        empty = False
        for name, data in [("train", train), ("val", val), ("test", test)]:
            elo_data = split_elo(data, elos, lo, hi)
            if len(elo_data) > 0:
                mmap = np.memmap(
                    os.path.join(tagdir, f"{name}.npy"),
                    mode="w+",
                    dtype="int64",
                    shape=elo_data.shape,
                )
                mmap[:] = elo_data[:]
                mmap.flush()
                fmd[name + "_shape"] = elo_data.shape
            else:
                empty = True
                with open(os.path.join(tagdir, "empty.txt"), "w") as f:
                    f.write("empty\n")

        if not empty:
            if not os.path.islink(os.path.join(tagdir, "md.npy")):
                os.symlink(
                    os.path.join(npydir, "md.npy"), os.path.join(tagdir, "md.npy")
                )
            if not os.path.islink(os.path.join(tagdir, "mvids.npy")):
                os.symlink(
                    os.path.join(npydir, "mvids.npy"),
                    os.path.join(tagdir, "mvids.npy"),
                )
            with open(os.path.join(tagdir, "fmd.json"), "w") as f:
                json.dump(fmd, f)


if __name__ == "__main__":
    main()
