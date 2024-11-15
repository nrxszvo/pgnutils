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
    return elos, train, val, test


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
    elos, train, val, test = load_data(args.npydir)
    elo_lo_hi = [0] + args.elo_edges + [float("inf")]
    for i in range(len(elo_lo_hi) - 1):
        lo = elo_lo_hi[i]
        hi = elo_lo_hi[i + 1]
        tag = f"{hi}" if hi != float("inf") else f"gt-{lo}"
        outdir = os.path.join(args.outdir, tag)
        os.makedirs(outdir, exist_ok=True)
        os.symlink(os.path.join(args.npydir, "md.npy"), os.path.join(outdir, "md.npy"))
        os.symlink(
            os.path.join(args.npydir, "mvids.npy"),
            os.path.join(outdir, "mvids.npy"),
        )
        for name, data in [("train", train), ("val", val), ("test", test)]:
            elo_data = split_elo(data, elos, lo, hi)
            if len(elo_data) > 0:
                mmap = np.memmap(
                    os.path.join(outdir, f"{name}.npy"),
                    mode="w+",
                    dtype="int64",
                    shape=elo_data.shape,
                )
                mmap[:] = elo_data[:]
                mmap.flush()
                with open(os.path.join(outdir, "fmd.json"), "w") as f:
                    json.dump({"dtype": "int64", "shape": elo_data.shape}, f)

            else:
                with open(os.path.join(outdir, "empty.txt"), "w") as f:
                    f.write("empty\n")


if __name__ == "__main__":
    main()
