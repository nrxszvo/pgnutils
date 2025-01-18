import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--npydir",
    nargs="+",
    help="list of numpy directories containing datasets to concatenate",
)
parser.add_argument("--outdir", help="output directory")


def concat(npydirs, outdir):
    md = {"archives": [], "ngames": 0, "nmoves": 0}
    for npydir in npydirs:
        submd = np.load(f"{npydir}/md.npy", allow_pickle=True).item()
        ngames = md["ngames"]
        nmoves = md["nmoves"]
        for name, ng, nm in submd["archives"]:
            md["archives"].append((name, ngames + ng, nmoves + nm))

        md["ngames"] += submd["ngames"]
        md["nmoves"] += submd["nmoves"]

    np.save(f"{outdir}/md.npy", md)
    welos = np.memmap(
        f"{outdir}/welos.npy", mode="w+", dtype="int16", shape=md["ngames"]
    )
    belos = np.memmap(
        f"{outdir}/belos.npy", mode="w+", dtype="int16", shape=md["ngames"]
    )
    gamestarts = np.memmap(
        f"{outdir}/gamestarts.npy", mode="w+", dtype="int64", shape=md["ngames"]
    )
    mvids = np.memmap(
        f"{outdir}/mvids.npy", mode="w+", dtype="int16", shape=md["nmoves"]
    )
    clk = np.memmap(f"{outdir}/clk.npy", mode="w+", dtype="int16", shape=md["nmoves"])

    ngames = 0
    nmoves = 0
    for npydir in npydirs:
        submd = np.load(f"{npydir}/md.npy", allow_pickle=True).item()
        subwelo = np.memmap(
            f"{npydir}/welos.npy", mode="r", dtype="int16", shape=submd["ngames"]
        )
        welos[ngames : ngames + submd["ngames"]] = subwelo[:]
        subbelo = np.memmap(
            f"{npydir}/belos.npy", mode="r", dtype="int16", shape=submd["ngames"]
        )
        belos[ngames : ngames + submd["ngames"]] = subbelo[:]
        subgs = np.memmap(
            f"{npydir}/gamestarts.npy", mode="r", dtype="int64", shape=submd["ngames"]
        )
        gamestarts[ngames : ngames + submd["ngames"]] = subgs[:] + nmoves
        submvid = np.memmap(
            f"{npydir}/mvids.npy", mode="r", dtype="int16", shape=submd["nmoves"]
        )
        mvids[nmoves : nmoves + submd["nmoves"]] = submvid[:]
        subclk = np.memmap(
            f"{npydir}/clk.npy", mode="r", dtype="int16", shape=submd["nmoves"]
        )
        clk[nmoves : nmoves + submd["nmoves"]] = subclk[:]
        ngames += submd["ngames"]
        nmoves += submd["nmoves"]


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    concat(args.npydir, args.outdir)
