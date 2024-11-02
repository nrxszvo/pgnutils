import argparse
import os
import re
import subprocess
import tempfile

import numpy as np
import wget
from multiprocessing import Queue, Process, Lock

from py.lib import timeit


def collect_existing_npy(npy_dir):
    existing = []
    procfn = os.path.join(npy_dir, "processed.txt")
    if os.path.exists(procfn):
        with open(procfn) as f:
            for line in f:
                vs = line.rstrip().split(",")
                if vs[-1] != "failed":
                    existing.append(vs[0])
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


def write_npys(tmpdir, npy_dir, npyname):
    def get_fn(name):
        return os.path.join(npy_dir, name)

    success = False
    nmoves = 0
    try:
        with open(get_fn("processed.txt"), "a") as f:
            f.write(f"{npyname},")

        elos = np.load(f"{tmpdir}/elos.npy")
        gamestarts = np.load(f"{tmpdir}/gamestarts.npy")
        moves = np.load(f"{tmpdir}/moves.npy")
        ngames = gamestarts.shape[0]
        nmoves = moves.shape[1]

        with open(get_fn("processed.txt"), "a") as f:
            f.write(f"{ngames},{nmoves}")

        if nmoves > 0:
            mdfile = get_fn("md.npy")
            elofile = get_fn("elos.npy")
            gsfile = get_fn("gamestarts.npy")
            mvfile = get_fn("moves.npy")

            exists = os.path.exists(mdfile)

            if exists:
                md = np.load(mdfile, allow_pickle=True).item()
            else:
                md = {"archives": [], "ngames": 0, "nmoves": 0}

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

            md["archives"].append((npyname, md["ngames"], md["nmoves"]))
            md["ngames"] += ngames
            md["nmoves"] += nmoves
            np.save(mdfile, md, allow_pickle=True)
        success = True
    finally:
        with open(get_fn("processed.txt"), "a") as f:
            if success:
                f.write("succeeded\n")
            else:
                f.write("failed\n")
    return nmoves


class PrintSafe:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, string, end="\n"):
        self.lock.acquire()
        try:
            print(string, end=end)
        finally:
            self.lock.release()


def download_proc(url_q, zst_q, print_safe):
    while True:
        url, npyname = url_q.get()
        if url == "DONE":
            break
        zst, _ = parse_url(url)
        if not os.path.exists(zst):
            if not os.path.exists(zst):
                print_safe(f"{npyname}: downloading...")
                _, time_str = timeit(
                    lambda: wget.download(url, bar=lambda a, b, c: None)
                )
                print_safe(f"{npyname}: finished downloading in {time_str}")
        zst_q.put((npyname, zst))


def start_download_proc(url_q, zst_q, print_safe):
    p = Process(target=download_proc, args=((url_q, zst_q, print_safe)))
    p.daemon = True
    p.start()
    return p


def main(list_fn, npy_dir, parser_bin):
    to_proc = collect_remaining(list_fn, npy_dir)
    if len(to_proc) == 0:
        print("All files already processed")
        return

    url_q = Queue()
    zst_q = Queue()

    print_safe = PrintSafe()
    dl_p = start_download_proc(url_q, zst_q, print_safe)
    url, npyname = to_proc[0]
    url_q.put((url, npyname))
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            for next_url, next_npy in to_proc[1:] + [("DONE", None)]:
                npyname, zst_fn = zst_q.get()
                url_q.put((next_url, next_npy))

                print_safe(f"{npyname}: processing zst...")
                cmd = [
                    "./" + parser_bin,
                    "--zst",
                    zst_fn,
                    "--name",
                    npyname,
                    "--outdir",
                    tempdir,
                ]
                p = subprocess.Popen(cmd)
                p.wait()

                nmoves = write_npys(tempdir, npy_dir, npyname)
                if nmoves == 0:
                    print("Last archive contained zero moves: terminating...")
                    break
                os.remove(zst_fn)
    finally:
        print_safe("closing main")
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
    parser.add_argument("--parser", default="processZst", help="parser binary")
    args = parser.parse_args()
    main(args.list, args.npy, args.parser)
