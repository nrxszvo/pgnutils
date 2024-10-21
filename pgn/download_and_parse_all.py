import argparse
import os
import re

import numpy as np
import pyzstd
import wget
from multiprocessing import Queue, Process, Lock

from lib import timeit
from parse_pgn import ParallelParser


def collect_existing_npy(npy_dir):
    existing = []
    for fn in os.listdir(npy_dir):
        m = re.match("lichess_([0-9\-]+)_md\.npy", fn)
        if m is not None:
            existing.append(m.group(1))
    return existing


def collect_remaining(list_fn, npy_dir):
    existing = collect_existing_npy(npy_dir)
    to_proc = []
    with open(list_fn) as f:
        for line in f:
            m = re.match(".+standard_rated_([0-9\-]+)\.pgn\.zst", line)
            if m is not None and m.group(1) not in existing:
                to_proc.append((line.rstrip(), m.group(1)))
    return to_proc


def parse_url(url):
    m = re.match(".*(lichess_db.*pgn\.zst)", url)
    zst = m.group(1)
    pgn_fn = zst[:-4]
    return zst, pgn_fn


def decompress(zst, pgn_fn):
    fin = open(zst, "rb")
    fout = open(pgn_fn, "wb")
    pyzstd.decompress_stream(fin, fout)
    fin.close()
    fout.close()


def write_npys(npy_dir, npyname, all_md, all_moves):
    if len(all_md["games"]) > 0:
        mdfile = f"{npy_dir}/lichess_{npyname}_md.npy"
        mvfile = f"{npy_dir}/lichess_{npyname}_moves.npy"
        all_md["shape"] = len(all_moves)
        np.save(mdfile, all_md, allow_pickle=True)
        output = np.memmap(mvfile, dtype="int32", mode="w+", shape=all_md["shape"])
        output[:] = all_moves[:]


class PrintSafe:
    def __init__(self, lock):
        self.lock = lock

    def __call__(self, string, end="\n"):
        self.lock.acquire()
        try:
            print(string, end=end)
        finally:
            self.lock.release()


def download_proc(url_q, zst_q, print_safe, save_intermediate):
    while True:
        url, npyname = url_q.get()
        if url == "DONE":
            break
        zst, pgn_fn = parse_url(url)
        if not os.path.exists(pgn_fn):
            if not os.path.exists(zst):
                print_safe(f"{npyname}: downloading...")
                _, time_str = timeit(
                    lambda: wget.download(url, bar=lambda a, b, c: None)
                )
                print_safe(f"{npyname}: finished downloading in {time_str}")

            print_safe(f"{npyname}: decompressing...")
            _, time_str = timeit(lambda: decompress(zst, pgn_fn))
            print_safe(f"{npyname}: finished decompressing in {time_str}")
            if not save_intermediate:
                os.remove(zst)

        zst_q.put((npyname, pgn_fn))


def start_download_proc(url_q, zst_q, print_safe, save_intermediate):
    p = Process(
        target=download_proc, args=((url_q, zst_q, print_safe, save_intermediate))
    )
    p.daemon = True
    p.start()
    return p


def main(list_fn, npy_dir, n_proc, save_intermediate):
    to_proc = collect_remaining(list_fn, npy_dir)
    url_q = Queue()
    zst_q = Queue()
    print_lock = Lock()
    pgn_parser = ParallelParser(n_proc)
    print_safe = PrintSafe(print_lock)

    dl_p = start_download_proc(url_q, zst_q, print_safe, save_intermediate)
    url, npyname = to_proc[0]
    url_q.put((url, npyname))
    try:
        for url, next_npy in to_proc[1:] + [("DONE", None)]:
            npyname, pgn_fn = zst_q.get()
            url_q.put((url, next_npy))

            print_safe(f"{npyname}: parsing pgn...")
            (all_md, all_moves), time_str = timeit(
                lambda: pgn_parser.parse(pgn_fn, npyname)
            )
            print_safe(f"{npyname}: finished parsing in {time_str}")
            print_safe(f"{npyname}: writing {len(all_md['games'])} games to file...")
            write_npys(npy_dir, npyname, all_md, all_moves)
            if not save_intermediate:
                os.remove(pgn_fn)

    finally:
        print_safe("closing main")
        pgn_parser.close()
        url_q.close()
        zst_q.close()
        try:
            dl_p.join(5)
        except Exception as e:
            print(e)
            dl_p.kill()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        default="list.txt",
        help="txt file containing list of pgn zips to download and parse",
    )
    parser.add_argument("--npy", default="npy", help="folder to save npy files")
    parser.add_argument("--save", action="store_true", help="save intermediate outputs")
    parser.add_argument(
        "--n_proc",
        default=os.cpu_count() - 1,
        help="number of reader processes",
        type=int,
    )
    args = parser.parse_args()
    main(args.list, args.npy, args.n_proc, args.save)
