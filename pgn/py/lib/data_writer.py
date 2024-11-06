import os
import numpy as np

from py.lib import timeit


class DataWriter:
    def __init__(self, outdir, alloc_games, alloc_moves):
        def get_fn(name):
            return os.path.join(outdir, name)

        self.processfn = get_fn("processed.txt")
        self.mdfile = get_fn("md.npy")
        self.welofile = get_fn("welos.npy")
        self.belofile = get_fn("belos.npy")
        self.gsfile = get_fn("gamestarts.npy")
        self.mvidfile = get_fn("mvids.npy")
        self.clkfile = get_fn("clk.npy")

        exists = os.path.exists(self.mdfile)
        if exists:
            self.md = np.load(self.mdfile, allow_pickle=True).item()
            alloc_games = self.md["games_alloc"]
            alloc_moves = self.md["moves_alloc"]
            mode = "r+"
        else:
            assert not os.path.exists(self.welofile)
            assert not os.path.exists(self.belofile)
            assert not os.path.exists(self.gsfile)
            assert not os.path.exists(self.mvidfile)
            assert not os.path.exists(self.clkfile)
            mode = "w+"

        self.all_welos = np.memmap(
            self.welofile, mode=mode, dtype="int16", shape=alloc_games
        )
        self.all_belos = np.memmap(
            self.belofile, mode=mode, dtype="int16", shape=alloc_games
        )
        self.all_gamestarts = np.memmap(
            self.gsfile, mode=mode, dtype="int64", shape=alloc_games
        )
        self.all_mvids = np.memmap(
            self.mvidfile, mode=mode, dtype="int16", shape=alloc_moves
        )
        self.all_clk = np.memmap(
            self.clkfile, mode=mode, dtype="int16", shape=alloc_moves
        )
        if not exists:
            print("allocating outputs...")

            def allocate():
                self.all_welos[:] = 0
                self.all_belos[:] = 0
                self.all_gamestarts[:] = 0
                self.all_mvids[:] = 0
                self.all_clk[:] = 0

            _, timestr = timeit(allocate)
            print(f"memory allocation took {timestr}")

            self.md = {
                "archives": [],
                "ngames": 0,
                "nmoves": 0,
                "games_alloc": alloc_games,
                "moves_alloc": alloc_moves,
            }
            np.save(self.mdfile, self.md, allow_pickle=True)

    def __del__(self):
        self.all_welos.flush()
        self.all_belos.flush()
        self.all_gamestarts.flush()
        self.all_mvids.flush()
        self.all_clk.flush()

    def write_npys(self, tmpdir, npyname):
        success = False
        nmoves = 0
        try:
            with open(self.processfn, "a") as f:
                f.write(f"{npyname},")

            elos = np.load(f"{tmpdir}/elos.npy")
            gamestarts = np.load(f"{tmpdir}/gamestarts.npy")
            moves = np.load(f"{tmpdir}/moves.npy")
            ngames = gamestarts.shape[0]
            nmoves = moves.shape[1]

            with open(self.processfn, "a") as f:
                f.write(f"{ngames},{nmoves},")

            if nmoves > 0:
                gs = self.md["ngames"]
                ge = gs + ngames
                ms = self.md["nmoves"]
                me = ms + nmoves

                def realloc(mmap, fn, dtype, cursize, shape):
                    newmap = np.memmap(".tmpmap", mode="w+", dtype=dtype, shape=shape)
                    newmap[:cursize] = mmap[:]
                    newmap[cursize:] = 0
                    newmap.flush()
                    os.remove(fn)
                    os.rename(".tmpmap", fn)
                    return newmap

                cursize = self.md["games_alloc"]
                if ge > cursize:
                    newsize = int(1.1 * cursize)

                    self.all_welos = realloc(
                        self.all_welos, self.welofile, "int16", cursize, newsize
                    )
                    self.all_belos = realloc(
                        self.all_belos, self.belofile, "int16", cursize, newsize
                    )
                    self.all_gamestarts = realloc(
                        self.all_gamestarts, self.gsfile, "int64", cursize, newsize
                    )
                    self.md["games_alloc"] = newsize

                cursize = self.md["moves_alloc"]
                if me > cursize:
                    newsize = int(1.1 * cursize)
                    self.all_mvids = realloc(
                        self.all_mvids, self.mvidfile, "int16", cursize, newsize
                    )
                    self.all_clk = realloc(
                        self.all_clk, self.clkfile, "int16", cursize, newsize
                    )
                    self.md["moves_alloc"] = newsize

                self.all_welos[gs:ge] = elos[0, :]
                self.all_belos[gs:ge] = elos[1, :]
                for i in range(ngames):
                    gamestarts[i] += self.md["nmoves"]
                self.all_gamestarts[gs:ge] = gamestarts[:]
                self.all_mvids[ms:me] = moves[0, :]
                self.all_clk[ms:me] = moves[1, :]

                self.md["archives"].append(
                    (npyname, self.md["ngames"], self.md["nmoves"])
                )
                self.md["ngames"] += ngames
                self.md["nmoves"] += nmoves
                np.save(self.mdfile, self.md, allow_pickle=True)
            success = True
        finally:
            with open(self.processfn, "a") as f:
                if success:
                    f.write("succeeded\n")
                else:
                    f.write("failed\n")
        return nmoves
