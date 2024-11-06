import argparse
import os
import sys
import re
import subprocess
import tempfile
import time

import wget
from multiprocessing import Queue, Process, Lock

from py.lib import timeit, DataWriter


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


class PrintSafe:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, string, end="\n"):
        self.lock.acquire()
        try:
            print(string, end=end)
        finally:
            self.lock.release()


def download_proc(pid, url_q, zst_q, print_safe):
    while True:
        url, npyname = url_q.get()
        if url == "DONE":
            zst_q.put(("DONE", None))
            break
        zst, _ = parse_url(url)
        if not os.path.exists(zst):
            if not os.path.exists(zst):
                while zst_q.qsize() > 2:
                    print_safe(f"download proc {pid} is sleeping...")
                    time.sleep(5 * 60)
                print_safe(f"{npyname}: downloading...")
                _, time_str = timeit(
                    lambda: wget.download(url, bar=lambda a, b, c: None)
                )
                print_safe(f"{npyname}: finished downloading in {time_str}")
        zst_q.put((npyname, zst))


def start_download_procs(url_q, zst_q, print_safe, nproc):
    procs = []
    for pid in range(nproc):
        p = Process(target=download_proc, args=((pid, url_q, zst_q, print_safe)))
        p.daemon = True
        p.start()
        procs.append(p)
    return procs


def main(
    list_fn,
    npy_dir,
    parser_bin,
    n_dl_proc,
    max_active_procs,
    n_reader_proc,
    n_move_proc,
    allow_no_clock,
    alloc_games,
    alloc_moves,
):
    to_proc = collect_remaining(list_fn, npy_dir)
    if len(to_proc) == 0:
        print("All files already processed")
        return

    dataWriter = DataWriter(npy_dir, alloc_games, alloc_moves)

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
                print_safe(f"{npyname}: processing zst into {tmpdir.name}...")
                cmd = [
                    "./" + parser_bin,
                    "--zst",
                    zst_fn,
                    "--name",
                    npyname,
                    "--outdir",
                    tmpdir.name,
                    "--nReaders",
                    str(n_reader_proc),
                    "--nMoveProcessors",
                    str(n_move_proc),
                ]
                if allow_no_clock:
                    cmd.append("--allowNoClock")

                p = subprocess.Popen(cmd)
                active_procs.append((p, tmpdir, npyname, zst_fn))

            def check_cleanup(p, tmpdir, name, zst):
                finished = False
                nmoves = None
                status = p.poll()
                if status is not None:
                    try:
                        if status != 0:
                            print_safe(f"{name}: poll returned {status}")
                            _, errs = p.communicate()
                            if errs is not None:
                                print_safe(f"{name}: returned errors:\n{errs}")
                            return True, 0

                        print_safe(f"{name}: writing to file from {tmpdir.name}...")
                        nmoves, timestr = timeit(
                            lambda: dataWriter.write_npys(tmpdir.name, name)
                        )
                        print_safe(f"{name}: finished writing in {timestr}")
                        os.remove(zst)
                        finished = True
                    finally:
                        tmpdir.cleanup()

                return finished, nmoves

            while len(active_procs) == max_active_procs or (
                terminate and len(active_procs) > 0
            ):
                time.sleep(5)
                for procdata in reversed(active_procs):
                    finished, nmoves = check_cleanup(*procdata)
                    if finished:
                        if nmoves == 0:
                            terminate = True
                            print_safe(
                                "Last archive contained no moves, 'terminate' signaled"
                            )
                        active_procs.remove(procdata)

            if terminate and len(active_procs) == 0:
                break

    finally:
        print_safe("cleaning up...")
        for p, tmpdir, _, zst in active_procs:
            p.kill()
            tmpdir.cleanup()
            # os.remove(zst)
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
        "--n_dl_procs",
        default=2,
        type=int,
        help="number of zsts to download in parallel",
    )
    parser.add_argument(
        "--n_active_procs",
        default=2,
        type=int,
        help="number of zsts to process in parallel",
    )
    parser.add_argument(
        "--n_reader_procs",
        default=2,
        help="number of decompressor/game parser threads",
        type=int,
    )
    parser.add_argument(
        "--n_move_procs",
        default=os.cpu_count() - 2,
        help="number of move parser threads",
        type=int,
    )

    parser.add_argument(
        "--allow_no_clock",
        default=False,
        action="store_true",
        help="allow games without clock time data to be included",
    )
    parser.add_argument(
        "--alloc_games",
        default=1024**3,
        help="initial memory allocation for game-level data (elos, gamestarts)",
        type=int,
    )
    parser.add_argument(
        "--alloc_moves",
        default=50 * 1024**3,
        help="initial memory allocation for move data (mvids, clk times)",
        type=int,
    )
    args = parser.parse_args()
    if args.alloc_moves > 1024**3:
        print(
            f"WARNING: allocating {4*args.alloc_moves/1024**3:.2f} GB of output.  Continue?"
        )
        resp = input("Y|n")
        if resp == "n":
            sys.exit()

    main(
        args.list,
        args.npy,
        args.parser,
        args.n_dl_procs,
        args.n_active_procs,
        args.n_reader_procs,
        args.n_move_procs,
        args.allow_no_clock,
        args.alloc_games,
        args.alloc_moves,
    )
