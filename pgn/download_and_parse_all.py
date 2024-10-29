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
    procfn = os.path.join(npy_dir, "processed.txt")
    if os.path.exists(procfn):
        with open(procfn) as f:
            existing = [name.rstrip() for name in f.readlines()]
    return existing


def collect_remaining(list_fn, npy_dir):
    existing = collect_existing_npy(npy_dir)
    to_proc = []
    with open(list_fn) as f:
        for line in f:
            npyname = re.match(".+standard_rated_([0-9\-]+)\.pgn\.zst", line).group(1)
            if npyname not in existing:
                to_proc.append((line.rstrip(), npyname))
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


def write_npys(npy_dir, npyname, ngames, nmoves, elos, gamestarts, moves):
    def get_fn(name):
        return os.path.join(npy_dir, name)

    with open(get_fn("processed.txt"), "a") as f:
        f.write(npyname + "\n")

    if nmoves == 0:
        return

    mdfile = get_fn("md.npy")
    elofile = get_fn("elos.npy")
    gsfile = get_fn("gamestarts.npy")
    mvfile = get_fn("moves.npy")

    exists = os.path.exists(mdfile)

    if exists:
        md = np.load(mdfile, allow_pickle=True).item()
    else:
        md = {"ngames": 0, "nmoves": 0}

    if exists:
        all_elos = np.load(elofile, allow_pickle=True)
    else:
        all_elos = np.empty((2, 0), dtype="int16")
    all_elos = np.concatenate([all_elos, elos[:, :ngames]], axis=1)
    np.save(elofile, all_elos, allow_pickle=True)
    del all_elos

    if exists:
        all_gamestarts = np.load(gsfile, allow_pickle=True)
    else:
        all_gamestarts = np.empty(0, dtype="int64")
    for i in range(ngames):
        gamestarts[i] += md["nmoves"]
    all_gamestarts = np.concatenate([all_gamestarts, gamestarts[:ngames]])
    np.save(gsfile, all_gamestarts, allow_pickle=True)
    del all_gamestarts

    if exists:
        all_moves = np.load(mvfile, allow_pickle=True)
    else:
        all_moves = np.empty((2, 0), dtype="int16")
    all_moves = np.concatenate([all_moves, moves[:, :nmoves]], axis=1)
    np.save(mvfile, all_moves, allow_pickle=True)
    del all_moves

    md["ngames"] += ngames
    md["nmoves"] += nmoves
    np.save(mdfile, md, allow_pickle=True)


class PrintSafe:
    def __init__(self):
        self.lock = Lock()

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
    if len(to_proc) == 0:
        print("All files already processed")
        return

    url_q = Queue()
    zst_q = Queue()

    print_safe = PrintSafe()
    pgn_parser = ParallelParser(n_proc, print_safe)

    dl_p = start_download_proc(url_q, zst_q, print_safe, save_intermediate)
    url, npyname = to_proc[0]
    url_q.put((url, npyname))
    try:
        for next_url, next_npy in to_proc[1:] + [("DONE", None)]:
            npyname, pgn_fn = zst_q.get()
            url_q.put((next_url, next_npy))

            print_safe(f"{npyname}: parsing pgn...")
            (ngames, nmoves, all_elos, gamestarts, all_moves), time_str = timeit(
                lambda: pgn_parser.parse(pgn_fn, npyname)
            )
            print_safe(f"{npyname}: finished parsing in {time_str}")
            print_safe(
                f"{npyname}: writing {ngames:.1e} games and {nmoves:.1e} moves to file..."
            )
            write_npys(
                npy_dir, npyname, ngames, nmoves, all_elos, gamestarts, all_moves
            )
            del all_elos
            del gamestarts
            del all_moves
            if not save_intermediate:
                os.remove(pgn_fn)

            if nmoves == 0:
                print("Last archive contained zero moves: terminating...")
                break

    finally:
        print_safe("closing main")
        pgn_parser.close()
        url_q.close()
        zst_q.close()
        try:
            dl_p.join(0.25)
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
    parser.add_argument("--npy", default="npy_w_clk", help="folder to save npy files")
    parser.add_argument("--save", action="store_true", help="save intermediate outputs")
    parser.add_argument(
        "--n_proc",
        default=os.cpu_count() - 1,
        help="number of reader processes",
        type=int,
    )
    args = parser.parse_args()
    main(args.list, args.npy, args.n_proc, args.save)
