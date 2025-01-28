import argparse
from py.lib import resize_mmaps

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--npydir", required=True, help="top-level folder containing block directories"
)
parser.add_argument(
    "--binary", default="resizeMMap", help="cpp binary for resizing files"
)
args = parser.parse_args()


if __name__ == "__main__":
    resize_mmaps(args.binary, args.npydir)
