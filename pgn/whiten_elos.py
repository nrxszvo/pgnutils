import numpy as np
import argparse
import json
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("fmd", help="fmd.json file with paths to block directories")
parser.add_argument(
    "--test",
    action="store_true",
    default=False,
    help="compare fast manual calculation to slower numpy calcualtion of params; dont save results",
)


def get_whiten_params_fast(fmd):
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

    return mean, std


def get_whiten_params_slow(fmd):
    elos = np.array([])
    for dn in fmd["block_dirs"]:
        for fn in ["welos.npy", "belos.npy"]:
            blk_elos = np.memmap(
                os.path.join(dn, fn),
                mode="r",
                dtype="int16",
            )
            elos = np.concatenate([elos, blk_elos])

    mean = elos.mean()
    std = elos.std()
    return mean, std


def main():
    args = parser.parse_args()
    with open(os.path.join(args.fmd)) as f:
        fmd = json.load(f)

    breakpoint()
    if args.test:
        test_mean, test_std = get_whiten_params_slow(fmd)
        print(f"test mean: {test_mean}")
        print(f"test std: {test_std}")

    mean, std = get_whiten_params_fast(fmd)

    print(f"mean: {mean}")
    print(f"std: {std}")

    if not args.test:
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
        with open(os.path.join(args.fmd), "w") as f:
            json.dump(fmd, f)


if __name__ == "__main__":
    main()
