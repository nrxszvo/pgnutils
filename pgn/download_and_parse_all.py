from parse_pgn import ParallelParser
import os
import re
import wget
import pyzstd
import numpy as np
import argparse
import time


def main(list_fn, npy_dir, n_proc, save_intermediate):
    existing = []
    for fn in os.listdir(npy_dir):
        m = re.match("lichess_([0-9\-]+)_md\.npy", fn)
        if m is not None:
            existing.append(m.group(1))

    to_proc = []
    with open(list_fn) as f:
        for line in f:
            m = re.match(".+standard_rated_([0-9\-]+)\.pgn\.zst", line)
            if m is not None and m.group(1) not in existing:
                to_proc.append((m.group(1), line.rstrip()))

    pgn_parser = ParallelParser(n_proc)
    for npyname, url in to_proc:
        start = time.time()
        m = re.match(".*(lichess_db.*pgn\.zst)", url)
        zst = m.group(1)
        pgn_fn = zst[:-4]

        if not os.path.exists(pgn_fn):
            if not os.path.exists(zst):
                print(f"downloading {url}")
                wget.download(url)

            print("\nunzipping...")
            fin = open(zst, "rb")
            fout = open(pgn_fn, "wb")
            pyzstd.decompress_stream(fin, fout)
            fin.close()
            fout.close()
            if not save_intermediate:
                os.remove(zst)

        print("parsing pgn...")
        all_md, all_moves = pgn_parser.parse(pgn_fn)

        print(f"\nNumber of games: {len(all_md['games'])}")
        if len(all_md["games"]) > 0:
            mdfile = f"npy/lichess_{npyname}_md.npy"
            mvfile = f"npy/lichess_{npyname}_moves.npy"
            all_md["shape"] = len(all_moves)
            np.save(mdfile, all_md, allow_pickle=True)
            output = np.memmap(mvfile, dtype="int32", mode="w+", shape=all_md["shape"])
            output[:] = all_moves[:]

        if not save_intermediate:
            os.remove(pgn_fn)
        end = time.time()
        nsec = end - start
        hr = int(nsec // 3600)
        minute = int((nsec % 3600) // 60)
        sec = int(nsec % 60)
        print(f"Total processing time: {hr}:{minute:02d}:{sec:02d}")
    pgn_parser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        default="list.txt",
        help="txt file containing list of pgn zips to download and parse",
    )
    parser.add_argument("--npy", default="npy", help="folder to support npy files")
    parser.add_argument("--save", action="store_true", help="save intermediate outputs")
    parser.add_argument(
        "--n_proc",
        default=os.cpu_count() - 1,
        help="number of reader processes",
        type=int,
    )
    args = parser.parse_args()
    main(args.list, args.npy, args.n_proc, args.save)
