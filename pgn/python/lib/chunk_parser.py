from . import PgnProcessor, parse_moves, validate_game, get_eta
from multiprocessing import Process, Queue, Lock
import os
import time


def process_chunk(pid, nproc, pgn_q, output_q, lock):
    while True:
        pgn = pgn_q.get()
        if pgn == "DONE":
            break
        else:
            nbytes = os.path.getsize(pgn)
            chunk_size = int(nbytes / nproc)
            byte_offset = pid * chunk_size
            if pid == nproc - 1:
                chunk_size = nbytes - byte_offset
            bytes_processed = 0
            gameid = 0

            with open(pgn) as fin:
                fin.seek(byte_offset)
                processor = PgnProcessor()
                for line in fin:
                    bytes_processed += len(line)
                    code = processor.process_line(line)
                    if code == "COMPLETE":
                        md = {
                            "WhiteElo": processor.get_welo(),
                            "BlackElo": processor.get_belo(),
                            "gameid": gameid,
                            "pid": pid,
                        }
                        gameid += 1
                        mvids = parse_moves(processor.get_move_str())
                        errs = validate_game(
                            md["gameid"], processor.get_move_str(), mvids
                        )
                        if len(errs) == 0:
                            output_q.put(("GAME", (pid, bytes_processed, md, mvids)))
                        else:
                            output_q.put(("ERROR", errs))
                    if (
                        code in ["COMPLETE", "INVALID"]
                        and bytes_processed >= chunk_size
                    ):
                        break

            output_q.put(("DONE", pid))


def start_chunk_reader_procs(num_readers, pgn_q, output_q, lock):
    procs = []
    for pid in range(num_readers):
        reader_p = Process(
            target=process_chunk, args=(pid, num_readers, pgn_q, output_q, lock)
        )
        reader_p.daemon = True
        reader_p.start()
        procs.append(reader_p)
    return procs


class ParallelChunkParser:
    def __init__(self, n_proc):
        self.n_proc = n_proc
        self.print_lock = Lock()
        self.pgn_q = Queue()
        self.output_q = Queue()
        self.reader_ps = start_chunk_reader_procs(
            n_proc, self.pgn_q, self.output_q, self.print_lock
        )

    def close(self):
        for _ in range(self.n_proc):
            self.pgn_q.put("DONE")
        self.output_q.close()
        self.pgn_q.close()
        for rp in self.reader_ps:
            rp.join()
            rp.close()

    def print_update(self):
        for pid in range(self.n_proc):
            eta_str = get_eta(self.chunk_size, self.pid_bp[pid], self.start)
            self.pid_etas[pid] = eta_str
        status_str = f"Parsed {self.total_games} games"
        for pid, (eta, bp) in enumerate(zip(self.pid_etas, self.pid_bp)):
            status_str += f"{eta} "
        self.print_lock.acquire()
        try:
            print(status_str, end="\r")
        finally:
            self.print_lock.release()

    def parse(self, pgn):
        self.start = time.time()
        self.nbytes = os.path.getsize(pgn)
        self.chunk_size = int(self.nbytes / self.n_proc)
        self.pid_etas = ["tbd"] * self.n_proc
        self.pid_bp = [0] * self.n_proc
        self.total_games = 0
        self.nmoves = [0] * self.n_proc

        for _ in range(self.n_proc):
            self.pgn_q.put(pgn)

        n_finished = 0
        all_pid_md = [[] for _ in range(self.n_proc)]
        all_pid_mvids = [[] for _ in range(self.n_proc)]

        while True:
            code, data = self.output_q.get()
            if code == "DONE":
                pid = data
                n_finished += 1
                self.print_update()
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
                self.pid_bp[pid] = bytesproc
                md["start"] = self.nmoves[pid]
                self.nmoves[pid] += len(mvids)
                md["end"] = self.nmoves[pid]
                all_pid_md[pid].append(md)
                all_pid_mvids[pid].append(mvids)
                self.total_games += 1
                if self.total_games % 1000 == 0:
                    self.print_update()
            else:
                raise Exception(f"invalid code: {code}")

        games = [md for pid_md in all_pid_md for md in pid_md]
        all_mvids = []
        for pid_mvids in all_pid_mvids:
            for mvids in pid_mvids:
                all_mvids.extend(mvids)

        all_md = {"shape": len(all_mvids), "games": games}
        return all_md, all_mvids
