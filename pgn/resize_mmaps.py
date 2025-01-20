import argparse
import numpy as np
import os
import subprocess

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--npydir", required=True, help="top-level folder containing block directories"
)
parser.add_argument(
    "--binary", default="resizeMMap", help="cpp binary for resizing files"
)
args = parser.parse_args()

if __name__ == "__main__":
    blockDirs = [dn for dn in os.listdir(args.npydir) if "block-" in dn]
    for dn in blockDirs:
        print(f"resizing {dn}")
        md = np.load(os.path.join(args.npydir, dn, "md.npy"), allow_pickle=True).item()
        cmd = [
            args.binary,
            "--blockDir",
            os.path.abspath(os.path.join(args.npydir, dn)),
            "--ngames",
            str(md["ngames"]),
            "--nmoves",
            str(md["nmoves"]),
        ]
        subprocess.call(cmd)
