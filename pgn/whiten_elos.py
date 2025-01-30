import numpy as np
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("datadir")
args = parser.parse_args()


def main():
    with open(os.path.join(args.datadir, "fmd.json")) as f:
        fmd = json.load(f)

    sumv = 0
    n = 0
    for dn in fmd["block_dirs"]:
        for fn in ["welos.npy", "belos.npy"]:
            elos = np.memmap(
                os.path.join(dn, fn),
                mode="r",
                dtype="int16",
            )
            sumv += elos.sum()
            n += elos.shape[0]
    mean = sumv / n

    sumv = 0
    for dn in fmd["block_dirs"]:
        for fn in ["welos.npy", "belos.npy"]:
            elos = np.memmap(
                os.path.join(dn, fn),
                mode="r",
                dtype="int16",
            )
            sumv += ((elos - mean) ** 2).sum()

    std = (sumv / (n - 1)) ** 0.5

    for dn in fmd["block_dirs"]:
        for fn in ["welos.npy", "belos.npy"]:
            elos = np.memmap(
                os.path.join(dn, fn),
                mode="r",
                dtype="int16",
            )
            whitened = (elos - mean) / std
            norm_elos = np.memmap(
                os.path.join(dn, f"whitened_{fn}"),
                mode="w+",
                dtype="float32",
                shape=elos.shape,
            )
            norm_elos[:] = whitened[:]

    fmd["elo_mean"] = mean
    fmd["elo_std"] = std
    with open(os.path.join(args.datadir, "fmd.json"), "w") as f:
        json.dump(fmd, f)


if __name__ == "__main__":
    main()
