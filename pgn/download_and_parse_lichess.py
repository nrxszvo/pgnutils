import argparse
import os
import re
import subprocess
import tempfile
import time

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
            f.write(f"{ngames},{nmoves},")

        def update_mmap(data, dtype, ndim, nitems, fn):
            mmap_update = np.memmap(
                ".tmpmap",
                mode="w+",
                dtype=dtype,
                shape=(ndim, nitems + data.shape[ndim - 1]),
            )
            if exists:
                mmap_old = np.memmap(fn, mode="r", dtype=dtype, shape=(ndim, nitems))
                mmap_update[:, :nitems] = mmap_old[:]
            mmap_update[:, nitems:] = data[:]
            os.rename(".tmpmap", fn)

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

            update_mmap(elos, "int16", 2, md["ngames"], elofile)
            for i in range(ngames):
                gamestarts[i] += md["nmoves"]
            update_mmap(gamestarts, "int64", 1, md["ngames"], gsfile)
            update_mmap(moves, "int16", 2, md["nmoves"], mvfile)

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
            zst_q.put(("DONE", None))
            break
        zst, _ = parse_url(url)
        if not os.path.exists(zst):
            if not os.path.exists(zst):
                while zst_q.qsize() > 2:
                    print_safe("download proc is sleeping...")
                    time.sleep(5 * 60)
                print_safe(f"{npyname}: downloading...")
                _, time_str = timeit(
                    lambda: wget.download(url, bar=lambda a, b, c: None)
                )
                print_safe(f"{npyname}: finished downloading in {time_str}")
        zst_q.put((npyname, zst))


def start_download_procs(url_q, zst_q, print_safe, nproc):
    procs = []
    for _ in range(nproc):
        p = Process(target=download_proc, args=((url_q, zst_q, print_safe)))
        p.daemon = True
        p.start()
        procs.append(p)
    return procs


def main(
    list_fn, npy_dir, parser_bin, nproc, allowNoClock, available_proc=os.cpu_count()
):
    n_dl_proc = int(available_proc / (nproc + 1))
    to_proc = collect_remaining(list_fn, npy_dir)
    if len(to_proc) == 0:
        print("All files already processed")
        return

    url_q = Queue()
    zst_q = Queue()

    print_safe = PrintSafe()
    dl_ps = start_download_procs(url_q, zst_q, print_safe, n_dl_proc)
    for url, name in to_proc:
        url_q.put((url, name))
    for _ in range(n_dl_proc):
        url_q.put(("DONE", None))

    n_dl_done = 0
    try:
        active_procs = []
        terminate = False
        while True:
            npyname, zst_fn = zst_q.get()
            if npyname == "DONE":
                n_dl_done += 1
                if n_dl_done == n_dl_proc:
                    terminate = True
            else:
                tmpdir = tempfile.TemporaryDirectory()
                print_safe(f"{npyname}: processing zst...")
                cmd = [
                    "./" + parser_bin,
                    "--zst",
                    zst_fn,
                    "--name",
                    npyname,
                    "--outdir",
                    tmpdir.name,
                    "--nReaders",
                    str(nproc),
                    "--allowNoClock",
                    str(allowNoClock),
                ]
                p = subprocess.Popen(cmd)
                active_procs.append((p, tmpdir, npyname, zst_fn))

            if len(active_procs) == 2 or (terminate and len(active_procs) > 0):
                to_remove = []
                while len(to_remove) == 0:
                    for i, (p, tmpdir, name, zst) in enumerate(active_procs):
                        if p.poll() is not None:
                            nmoves = write_npys(tmpdir.name, npy_dir, name)
                            if nmoves == 0:
                                print(
                                    "Last archive contained zero moves: terminating..."
                                )
                                terminate = True
                            to_remove.append(i)
                            os.remove(zst)
                            tmpdir.cleanup()

                for i in to_remove:
                    active_procs.remove(active_procs[i])

            if terminate and len(active_procs) == 0:
                break

    finally:
        print_safe("cleaning up...")
        url_q.close()
        zst_q.close()
        for dl_p in dl_ps:
            try:
                dl_p.join(0.25)
            except Exception as e:
                print(e)
                dl_p.kill()
        for fn in os.listdir("."):
            if re.match("lichess_db_standard_rated.*\.zst.*\.tmp", fn):
                os.remove(fn)


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
    parser.add_argument(
        "--nproc",
        default=os.cpu_count() - 1,
        help="number of parallel parser threads",
        type=int,
    )
    parser.add_argument(
        "--allowNoClock",
        default=False,
        action="store_true",
        help="allow games without clock time data to be included",
    )
    args = parser.parse_args()
    main(args.list, args.npy, args.parser, args.nproc, args.allowNoClock)
