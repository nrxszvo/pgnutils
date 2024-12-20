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

    elos = np.memmap(
        os.path.join(args.datadir, "elo.npy"),
        mode="r",
        dtype="int16",
        shape=(fmd["ngames"], 2),
    )
    mean = np.mean(elos)
    std = np.std(elos)
    whitened = (elos - mean) / std
    norm_elos = np.memmap(
        os.path.join(args.datadir, "whitened_elos.npy"),
        mode="w+",
        dtype="float32",
        shape=(fmd["ngames"], 2),
    )
    norm_elos[:] = whitened[:]
    fmd["elo_mean"] = mean
    fmd["elo_std"] = std
    with open(os.path.join(args.datadir, "fmd.json"), "w") as f:
        json.dump(fmd, f)


if __name__ == "__main__":
    main()
