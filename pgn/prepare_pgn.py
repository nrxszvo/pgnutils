import subprocess
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pgn", help="pgn file to process")
parser.add_argument("outdir", help="output directory for npy files")
parser.add_argument("--zst_binary", default="./processZst", help="processZst binary")
parser.add_argument("--filter_binary", default="./filterData", help="filterData binary")


def prepare_pgn(pgn, outdir, zst_bin, filt_bin):
    cmd = ["pzstd", pgn]
    subprocess.call(cmd)
    blkdir = os.path.join(outdir, "block-0")
    os.makedirs(blkdir, exist_ok=True)
    cmd = [zst_bin, "--zst", f"{pgn}.zst", "--serial", "--outdir", blkdir]
    subprocess.call(cmd)
    os.remove(f"{pgn}.zst")

    # convert to memmaps
    md = {}
    for fn in ["elos.npy", "gamestarts.npy", "moves.npy", "timeData.npy"]:
        fullfn = os.path.join(blkdir, fn)
        data = np.load(fullfn)
        if fn == "elos.npy":
            md["ngames"] = data.shape[1]
            nms = ["welos.npy", "belos.npy"]
        elif fn == "moves.npy":
            md["nmoves"] = data.shape[1]
            nms = ["mvids.npy", "clk.npy"]
        elif fn == "timeData.npy":
            nms = ["timeCtl.npy", "inc.npy"]
        else:
            nms = [fn]

        for i, nm in enumerate(nms):
            mmap = np.memmap(
                os.path.join(blkdir, nm),
                mode="w+",
                dtype="int64" if fn == "gamestarts.npy" else "int16",
                shape=data.shape[-1],
            )
            if data.ndim == 1:
                mmap[:] = data[:]
            else:
                mmap[:] = data[i, :]
    np.save(os.path.join(blkdir, "md.npy"), md, allow_pickle=True)
    for fn in ["elos.npy", "moves.npy", "timeData.npy"]:
        os.remove(os.path.join(blkdir, fn))

    filtdir = os.path.join(outdir, "filtered")
    os.makedirs(filtdir, exist_ok=True)
    cmd = [
        filt_bin,
        "--npydir",
        outdir,
        "--outdir",
        filtdir,
        "--trainp",
        "0",
        "--testp",
        "1",
    ]
    subprocess.call(cmd)


if __name__ == "__main__":
    args = parser.parse_args()
    prepare_pgn(args.pgn, args.outdir, args.zst_binary, args.filter_binary)
