import os
import numpy as np
from time import gmtime, strftime


def now():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


class DataWriter:
    def _get_last_block(self):
        block = 0
        dn = os.path.join(self.outdir, f"block-{block}")
        if not os.path.exists(dn):
            return block

        while os.path.exists(dn):
            block += 1
            dn = os.path.join(self.outdir, f"block-{block}")

        dn = os.path.join(self.outdir, f"block-{block - 1}")
        if os.path.exists(dn):
            md = np.load(os.path.join(dn, "md.npy"), allow_pickle=True).item()
            if (md["games_alloc"] - md["ngames"]) > 0 and (
                md["moves_alloc"] - md["nmoves"]
            ) > 0:
                return block - 1
        return block

    def _init_block(self, block, exist_ok=True):
        dn = os.path.join(self.outdir, f"block-{block}")
        os.makedirs(dn, exist_ok=exist_ok)

        def get_fn(name):
            return os.path.join(dn, name)

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
            alloc_games = self.alloc_games
            alloc_moves = self.alloc_moves
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
            self.md = {
                "archives": [],
                "ngames": 0,
                "nmoves": 0,
                "games_alloc": alloc_games,
                "moves_alloc": alloc_moves,
            }
            np.save(self.mdfile, self.md, allow_pickle=True)
        return block

    def __init__(self, outdir, alloc_games, alloc_moves):
        self.outdir = outdir
        self.alloc_games = alloc_games
        self.alloc_moves = alloc_moves
        self.block = self._get_last_block()
        self._init_block(self.block)
        self.processfn = os.path.join(outdir, "processed.txt")

    def _flush_all(self):
        try:
            self.all_welos.flush()
            self.all_belos.flush()
            self.all_gamestarts.flush()
            self.all_mvids.flush()
            self.all_clk.flush()
        except Exception as e:
            print(f"Warning: attempting to flush memmaps raised exception: {e}")

    def __del__(self):
        self._flush_all()

    def write_npys(self, tmpdir, npyname):
        success = False
        nmoves = 0
        try:
            with open(self.processfn, "a") as f:
                f.write(f"[{now()}],{npyname},")

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

                if ge > self.md["games_alloc"] or me > self.md["moves_alloc"]:
                    print(f"initializing block {self.block + 1}")
                    self._flush_all()
                    self.block = self._init_block(self.block + 1)
                    gs = 0
                    ge = ngames
                    ms = 0
                    me = nmoves

                with open(self.processfn, "a") as f:
                    f.write(f"b{self.block},")

                self.all_welos[gs:ge] = elos[0, :]
                self.all_belos[gs:ge] = elos[1, :]
                gamestarts += self.md["nmoves"]
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
