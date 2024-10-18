from parse_pgn import main_parallel
import os
import re
import wget
import pyzstd
import linecache
import tracemalloc
import numpy as np
import argparse


def display_snap(snapshot, key_type="lineno", limit=10):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def main(list_fn, npy_dir):
    # tracemalloc.start()
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

    for npyname, url in to_proc:
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
            os.remove(zst)

        print("parsing pgn...")
        all_md, all_moves = main_parallel(pgn_fn, os.cpu_count() - 1)

        print(f"\nNumber of games: {len(all_md['games'])}")
        if len(all_md["games"]) > 0:
            mdfile = f"npy/lichess_{npyname}_md.npy"
            mvfile = f"npy/lichess_{npyname}_moves.npy"
            all_md["shape"] = len(all_moves)
            np.save(mdfile, all_md, allow_pickle=True)
            output = np.memmap(mvfile, dtype="int32", mode="w+", shape=all_md["shape"])
            output[:] = all_moves[:]

        os.remove(pgn_fn)
        # snapshot = tracemalloc.take_snapshot()
        # display_snap(snapshot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        default="list.txt",
        help="txt file containing list of pgn zips to download and parse",
    )
    parser.add_argument("--npy", default="npy", help="folder to support npy files")
    args = parser.parse_args()
    main(args.list, args.npy)
