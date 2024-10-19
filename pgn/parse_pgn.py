import argparse
import numpy as np
import os
import time
import datetime
from multiprocessing import Process, Queue, Lock
import traceback
from lib import validate_game, parse_moves, PgnProcessor


def get_eta(max_items, items_so_far, start):
    end = time.time()
    eta = datetime.timedelta(
        seconds=(max_items - items_so_far) * (end - start) / items_so_far
    )
    hours = eta.seconds // 3600
    minutes = (eta.seconds % 3600) // 60
    seconds = eta.seconds % 60
    eta_str = f"{eta.days}:{hours}:{minutes:02}:{seconds:02}"
    return eta_str


def load_games(pgn_q, games_q, num_readers, lock):
    while True:
        pgn = pgn_q.get()
        if pgn != "DONE":
            bytes_processed = 0
            gameid = 0
            with open(pgn) as fin:
                processor = PgnProcessor()
                while True:
                    for line in fin:
                        bytes_processed += len(line)
                        code = processor.process_line(line)
                        if code == "COMPLETE":
                            md = {
                                "WhiteElo": processor.get_welo(),
                                "BlackElo": processor.get_belo(),
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
                                    ),
                                )
                            )

                    else:
                        break  # EOF

        if pgn == "DONE":
            for _ in range(num_readers):
                games_q.put(("SESSION_DONE", None))
            break
        else:
            for _ in range(num_readers):
                games_q.put(("FILE_DONE", None))


def start_games_reader(pgn_q, games_q, n_proc, output_q):
    games_p = Process(target=load_games, args=(pgn_q, games_q, n_proc, output_q))
    games_p.daemon = True
    games_p.start()
    return games_p


def process_games(games_q, output_q, pid, lock):
    while True:
        code, data = games_q.get()
        if code == "FILE_DONE":
            output_q.put(("DONE", None))
        elif code == "SESSION_DONE":
            break

        else:
            bytesproc, md, move_str = data
            mvids = parse_moves(move_str)
            errs = validate_game(md["gameid"], move_str, mvids)
            if len(errs) == 0:
                output_q.put(("GAME", (pid, bytesproc, md, mvids)))
            else:
                output_q.put(("ERROR", errs))


def start_reader_procs(num_readers, games_q, output_q, lock):
    procs = []
    for pid in range(num_readers):
        reader_p = Process(target=process_games, args=(games_q, output_q, pid, lock))
        reader_p.daemon = True
        reader_p.start()
        procs.append(reader_p)
    return procs


class ParallelParser:
    def __init__(self, n_proc):
        self.n_proc = n_proc
        self.print_lock = Lock()
        self.pgn_q = Queue()
        self.games_q = Queue()
        self.output_q = Queue()
        self.reader_ps = start_reader_procs(
            n_proc, self.games_q, self.output_q, self.print_lock
        )
        self.game_p = start_games_reader(
            self.pgn_q, self.games_q, n_proc, self.print_lock
        )

    def close(self):
        self.pgn_q.put("DONE")
        self.games_q.close()
        self.output_q.close()
        self.pgn_q.close()
        for rp in self.reader_ps:
            rp.join()
            rp.close()
        self.game_p.join()
        self.game_p.close()

    def parse(self, pgn):
        start = time.time()
        nbytes = os.path.getsize(pgn)
        self.pgn_q.put(pgn)
        n_finished = 0
        games = []
        all_mvids = []
        pid_counts = [0] * self.n_proc
        while True:
            code, data = self.output_q.get()
            if code == "DONE":
                n_finished += 1
                if n_finished == self.n_proc:
                    break
            elif code == "ERROR":
                self.print_lock.acquire()
                try:
                    print()
                    for err in data:
                        print(err)
                finally:
                    self.print_lock.release()
            elif code == "GAME":
                pid, bytesproc, md, mvids = data
                pid_counts[pid] += 1
                nmoves = len(all_mvids)
                md["start"] = nmoves
                md["end"] = nmoves + len(mvids)
                games.append(md)
                all_mvids.extend(mvids)

                if md["gameid"] % 1 == 0:
                    eta_str = get_eta(nbytes, bytesproc, start)
                    status_str = f"parsed {md['gameid']} games (eta: {eta_str})        "
                    self.print_lock.acquire()
                    try:
                        print(status_str, end="\r")
                    finally:
                        self.print_lock.release()

            else:
                raise Exception(f"invalid code: {code}")

        end = time.time()
        nsec = end - start
        hr = int(nsec // 3600)
        minute = int((nsec % 3600) // 60)
        sec = int(nsec % 60)
        print(
            f"Total time to parse pgn: {hr}:{minute:02d}:{sec:02d}                       "
        )
        print(f"pid counts: {pid_counts}")
        all_md = {"shape": len(all_mvids), "games": games}
        return all_md, all_mvids


def main_serial(pgn_file):
    # for debugging
    lineno = 0
    gamestart = 0

    # info
    game = 0
    bytes_processed = 0

    all_moves = []
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
            data.append(line)
            lineno += 1
            try:
                code = processor.process_line(line)
            except Exception as e:
                print(traceback.format_exc())
                print(e)
            if code == "COMPLETE":
                try:
                    mvids = parse_moves(processor.get_move_str())
                    errs = validate_game(gamestart, processor.get_move_str(), mvids)
                    if len(errs) > 0:
                        for err in errs:
                            print(err)
                        raise Exception("evaluation failed")

                    md["games"].append(
                        {
                            "WhiteElo": processor.get_welo(),
                            "BlackElo": processor.get_belo(),
                            "start": nmoves,
                            "length": len(mvids),
                        }
                    )
                    nmoves += len(mvids)
                    all_moves.extend(mvids)
                    game += 1
                    if game % 100 == 0:
                        eta_str = get_eta(nbytes, bytes_processed, start)
                        print(f"processed {game} games (eta: {eta_str})", end="\r")

                except Exception as e:
                    print(e)
                    print(f"game start: {gamestart}")

                gamestart = lineno + 1

        else:
            break  # EOF

        fin.close()

    return md, all_moves


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
        all_md, all_moves = main_serial(args.pgn)
    else:
        parser = ParallelParser(args.n_proc)
        all_md, all_moves = parser.parse(args.pgn)

    print(f"\nNumber of games: {len(all_md['games'])}")
    if len(all_md["games"]) > 0:
        mdfile = f"{args.npy}_md.npy"
        mvfile = f"{args.npy}_moves.npy"
        all_md["shape"] = len(all_moves)
        np.save(mdfile, all_md, allow_pickle=True)
        output = np.memmap(mvfile, dtype="int32", mode="w+", shape=all_md["shape"])
        output[:] = all_moves[:]


if __name__ == "__main__":
    main()
