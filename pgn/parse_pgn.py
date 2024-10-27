import argparse
import os
import time
import datetime
import traceback
from multiprocessing import Lock, Process, Queue

import numpy as np

from lib import PgnProcessor, parse_moves, validate_game, get_eta


def load_games(pgn_q, games_q, num_readers, lock):
    while True:
        pgn = pgn_q.get()
        if pgn == "DONE":
            for _ in range(num_readers):
                games_q.put(("SESSION_DONE", None))
            break
        else:
            bytes_processed = 0
            gameid = 0
            gamestart = 0
            with open(pgn) as fin:
                processor = PgnProcessor()
                for i, line in enumerate(fin):
                    bytes_processed += len(line)
                    code = processor.process_line(line)
                    if code == "COMPLETE":
                        md = {
                            "WhiteElo": processor.get_welo(),
                            "BlackElo": processor.get_belo(),
                            "time": processor.get_time(),
                            "gameid": gameid,
                        }
                        gameid += 1
                        games_q.put(
                            (
                                "GAME",
                                (
                                    bytes_processed,
                                    md,
                                    processor.get_move_str(),
                                    f"{pgn}:{gamestart}",
                                ),
                            )
                        )
                        gamestart = i + 1

            for _ in range(num_readers):
                games_q.put(("FILE_DONE", gameid))


def start_games_reader(pgn_q, games_q, n_proc, output_q):
    games_p = Process(target=load_games, args=(pgn_q, games_q, n_proc, output_q))
    games_p.daemon = True
    games_p.start()
    return games_p


def process_games(games_q, output_q, pid, lock, session_id):
    while True:
        code, data = games_q.get()
        if code in ["FILE_DONE", "SESSION_DONE"]:
            output_q.put(("DONE", data))
            if code == "SESSION_DONE":
                break

        else:
            bytesproc, md, move_str, dbg_info = data
            try:
                mvids, clk = parse_moves(move_str)
                errs = validate_game(md["gameid"], move_str, mvids)
                if len(errs) == 0:
                    output_q.put(("GAME", (pid, bytesproc, md, mvids, clk)))
                else:
                    output_q.put(("ERROR", errs))
            except Exception as e:
                with open(f"{session_id}_errs.txt", "a") as ferr:
                    ferr.write(f"{dbg_info}\n")
                    ferr.write(str(md) + "\n")
                    ferr.write(str(e) + "\n\n")
                output_q.put(("INVALID", bytesproc))


def start_reader_procs(num_readers, games_q, output_q, lock, session_id):
    procs = []
    for pid in range(num_readers):
        reader_p = Process(
            target=process_games, args=(games_q, output_q, pid, lock, session_id)
        )
        reader_p.daemon = True
        reader_p.start()
        procs.append(reader_p)
    return procs


class ParallelParser:
    def __init__(self, n_proc, session_id=datetime.datetime.now().isoformat()):
        self.n_proc = n_proc
        self.print_lock = Lock()
        self.pgn_q = Queue()
        self.games_q = Queue()
        self.output_q = Queue()
        self.session_id = session_id
        self.reader_ps = start_reader_procs(
            n_proc, self.games_q, self.output_q, self.print_lock, self.session_id
        )
        self.game_p = start_games_reader(
            self.pgn_q, self.games_q, n_proc, self.print_lock
        )

    def close(self):
        self.pgn_q.put("DONE")
        for _ in range(self.n_proc):
            self.output_q.get()
        self.games_q.close()
        self.output_q.close()
        self.pgn_q.close()
        for rp in [self.game_p] + self.reader_ps:
            try:
                rp.join(0.1)
            except Exception as e:
                print(e)
                rp.kill()

    def parse(self, pgn, name):
        nbytes = os.path.getsize(pgn)
        self.pgn_q.put(pgn)
        games = []
        all_mvids = []
        all_clk = []
        max_bp = 0
        ngames = 0
        prog = 0
        total_games = float("inf")
        n_finished = 0
        nmoves = 0
        start = time.time()
        while ngames < total_games or n_finished < self.n_proc:
            code, data = self.output_q.get()
            if code == "DONE":
                total_games = data
                n_finished += 1
            elif code == "ERROR":
                self.print_lock.acquire()
                try:
                    print()
                    for err in data:
                        print(err)
                finally:
                    self.print_lock.release()
            elif code == "INVALID":
                ngames += 1
            elif code == "GAME":
                pid, bytesproc, md, mvids, clk = data
                max_bp = max(max_bp, bytesproc)
                ngames += 1

                md["start"] = nmoves
                nmoves += len(mvids)
                md["end"] = nmoves
                games.append(md)
                all_mvids.extend(mvids)
                all_clk.extend(clk)

                total_games_est = ngames / (bytesproc / nbytes)
                cur_prog = int(10 * ngames / total_games_est)
                if cur_prog > prog:
                    prog = cur_prog
                    eta_str = get_eta(nbytes, max_bp, start)
                    status_str = f"{name}: parsed {ngames} games ({10*prog}% done, eta: {eta_str})"
                    self.print_lock.acquire()
                    try:
                        print(status_str)
                    finally:
                        self.print_lock.release()

            else:
                raise Exception(f"invalid code: {code}")

        all_md = {"shape": nmoves, "games": games}
        return all_md, all_mvids, all_clk


def main_serial(pgn_file):
    # for debugging
    lineno = 0
    gamestart = 0

    # info
    game = 0
    bytes_processed = 0

    all_moves = []
    all_clk = []
    md = {"games": []}
    nbytes = os.path.getsize(pgn_file)
    fin = open(pgn_file, "r")

    start = time.time()
    nmoves = 0
    processor = PgnProcessor()
    while True:
        data = []
        for i, line in enumerate(fin):
            bytes_processed += len(line)
            print(f"{100*bytes_processed/nbytes:.1f}% done... ({game} games)", end="\r")
            data.append(line)
            lineno += 1
            try:
                code = processor.process_line(line)
            except Exception as e:
                print(traceback.format_exc())
                print(e)
            if code == "COMPLETE":
                try:
                    mvids, clk = parse_moves(processor.get_move_str())
                    errs = validate_game(gamestart, processor.get_move_str(), mvids)
                    if len(errs) > 0:
                        for err in errs:
                            print(err)
                        raise Exception("evaluation failed")

                    md["games"].append(
                        {
                            "WhiteElo": processor.get_welo(),
                            "BlackElo": processor.get_belo(),
                            "time": processor.get_time(),
                            "start": nmoves,
                            "length": len(mvids),
                        }
                    )
                    nmoves += len(mvids)
                    all_moves.extend(mvids)
                    all_clk.extend(clk)
                    game += 1
                    if game % 100 == 0:
                        eta_str = get_eta(nbytes, bytes_processed, start)
                        print(f"processed {game} games (eta: {eta_str})", end="\r")

                except Exception as e:
                    print(e)
                    print(f"game start: {gamestart}")

                gamestart = lineno + 1

            if code in ["COMPLETE", "INVALID"]:
                data = []

        else:
            break  # EOF

        fin.close()

    return md, all_moves, all_clk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", help="pgn filename", required=True)
    parser.add_argument("--npy", help="npy output name", required=True)
    parser.add_argument(
        "--n_proc",
        help="number of reader processes",
        default=os.cpu_count() - 1,
        type=int,
    )
    parser.add_argument(
        "--serial", help="run in single process", action="store_true", default=False
    )

    args = parser.parse_args()

    if args.serial:
        all_md, all_moves, all_clk = main_serial(args.pgn)
    else:
        parser = ParallelParser(args.n_proc)
        all_md, all_moves, all_clk = parser.parse(args.pgn, os.path.basename(args.npy))
        parser.close()

    if len(all_md["games"]) > 0:
        mdfile = f"{args.npy}_md.npy"
        mvfile = f"{args.npy}_moves.npy"
        clkfile = f"{args.npy}_clk.npy"
        all_md["shape"] = len(all_moves)
        np.save(mdfile, all_md, allow_pickle=True)
        output = np.memmap(mvfile, dtype="int32", mode="w+", shape=all_md["shape"])
        output[:] = all_moves[:]
        output = np.memmap(clkfile, dtype="int32", mode="w+", shape=all_md["shape"])
        output[:] = all_clk[:]


if __name__ == "__main__":
    import multiprocessing

    __spec__ = None
    multiprocessing.set_start_method("spawn")
    main()
