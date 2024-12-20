import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import json
from pgn import NOOP


def init_worker(seed):
    np.random.seed(seed)


def collate_fn(batch):
    mininp = float("inf")
    openmoves = batch[0]["opening"].shape[0]
    for d in batch:
        mininp = min(mininp, d["n_inp"])

    mintgt = mininp + 1 - openmoves
    bs = len(batch)
    inputs = torch.full((bs, mininp), NOOP, dtype=torch.int32)
    openings = torch.empty((bs, openmoves), dtype=torch.int64)
    targets = torch.full((bs, mintgt), NOOP, dtype=torch.float32)

    for i, d in enumerate(batch):
        inp = d["input"]
        tgt = d["target"]
        inputs[i] = torch.from_numpy(inp[:mininp])
        targets[i] = torch.from_numpy(tgt[:mintgt])
        openings[i] = torch.from_numpy(d["opening"])

    return {
        "input": inputs,
        "target": targets,
        "opening": openings,
    }


def cheat_collate_fn(batch):
    maxinp = 0
    maxtgt = 0
    maxcd = 3
    openmoves = batch[0]["opening"].shape[0]
    for d in batch:
        inp = d["input"]
        tgt = d["w_target"]
        maxinp = max(maxinp, inp.shape[0])
        maxtgt = max(maxtgt, tgt.shape[0])
        maxcd = max(maxcd, len(d["cheatdata"]))

    bs = len(batch)
    inputs = torch.full((bs, maxinp), NOOP, dtype=torch.int32)
    cheatdata = -torch.ones((bs, maxcd, 2), dtype=torch.int64)
    openings = torch.empty((bs, openmoves), dtype=torch.int64)
    elos = torch.empty((bs, 2), dtype=torch.int64)
    wtargets = torch.full((bs, maxtgt), NOOP, dtype=torch.int64)
    btargets = torch.full((bs, maxtgt), NOOP, dtype=torch.int64)
    heads = torch.empty((bs, 2), dtype=torch.int64)
    offset_heads = torch.empty((bs, 2), dtype=torch.int64)

    nhead = batch[0]["nhead"]
    for i, d in enumerate(batch):
        inp = d["input"]
        wtgt = d["w_target"]
        btgt = d["b_target"]
        ni = inp.shape[0]
        nt = wtgt.shape[0]
        inputs[i, :ni] = torch.from_numpy(inp)
        ncd = len(d["cheatdata"])
        if ncd > 0:
            cheatdata[i, :ncd] = torch.from_numpy(d["cheatdata"])
        wtargets[i, :nt] = torch.from_numpy(wtgt)
        btargets[i, :nt] = torch.from_numpy(btgt)
        openings[i] = torch.from_numpy(d["opening"])
        elos[i] = d["elos"]
        heads[i, 0] = d["w_head"]
        heads[i, 1] = d["b_head"]
        offset_heads[i, 0] = i * nhead + heads[i, 0]
        offset_heads[i, 1] = i * nhead + heads[i, 1]

    return {
        "input": inputs,
        "cheatdata": cheatdata,
        "w_target": wtargets,
        "b_target": btargets,
        "opening": openings,
        "elos": elos,
        "heads": heads,
        "offset_heads": offset_heads,
    }


class MMCDataset(Dataset):
    def __init__(
        self,
        seq_len,
        opening_moves,
        indices,
        mvids,
        elos,
        elo_edges,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.opening_moves = opening_moves
        self.indices = indices
        self.nsamp = len(self.indices)
        self.mvids = mvids
        self.elos = elos
        self.elo_edges = elo_edges

    def __len__(self):
        return self.nsamp

    def _get_group(self, elo):
        for i, edge in enumerate(self.elo_edges):
            if elo <= edge:
                return i

    def __getitem__(self, idx):
        gs, nmoves, gidx = self.indices[idx]
        n_inp = min(self.seq_len, nmoves)
        inp = np.empty(n_inp, dtype="int32")
        inp[:] = self.mvids[gs : gs + n_inp]

        opening = np.empty(self.opening_moves, dtype="int64")
        opening[:] = self.mvids[gs : gs + self.opening_moves]

        tgt = np.empty(n_inp + 1 - self.opening_moves, dtype="float32")
        # tgt[:] = self.mvids[gs + self.opening_moves : gs + n_inp + 1]
        welo, belo = self.elos[gidx]
        tgt[::2] = welo
        tgt[1::2] = belo

        return {
            "input": inp,
            "target": tgt,
            "opening": opening,
            "n_inp": n_inp,
        }


class MMCCheatingDataset(Dataset):
    def __init__(
        self, seq_len, opening_moves, indices, cheatdata, mvids, elos, elo_edges
    ):
        super().__init__()
        self.seq_len = seq_len
        self.opening_moves = opening_moves
        self.nsamp = len(indices)
        self.indices = indices
        self.cheatdata = {}
        for _, nmoves, gidx in indices:
            cd = cheatdata[gidx]
            filtered = []
            for offset, cheat_mvid, _, _ in cd:
                if offset >= opening_moves and offset <= min(seq_len, nmoves - 1):
                    filtered.append([offset - opening_moves, cheat_mvid])
            self.cheatdata[gidx] = np.array(filtered)
        self.mvids = mvids
        self.elos = elos
        self.elo_edges = elo_edges

    def __len__(self):
        return self.nsamp

    def _get_group(self, elo):
        for i, edge in enumerate(self.elo_edges):
            if elo <= edge:
                return i

    def __getitem__(self, idx):
        gs, nmoves, gidx = self.indices[idx]
        welo, belo = self.elos[gidx]
        n_inp = min(self.seq_len, nmoves - 1)
        inp = np.empty(n_inp, dtype="int32")
        inp[:] = self.mvids[gs : gs + n_inp]

        opening = np.empty(self.opening_moves, dtype="int64")
        opening[:] = self.mvids[gs : gs + self.opening_moves]

        w_tgt = np.empty(n_inp + 1 - self.opening_moves, dtype="int64")
        b_tgt = np.empty(n_inp + 1 - self.opening_moves, dtype="int64")
        w_tgt[:] = self.mvids[gs + self.opening_moves : gs + n_inp + 1]
        b_tgt[:] = self.mvids[gs + self.opening_moves : gs + n_inp + 1]
        w_tgt[1::2] = NOOP
        b_tgt[::2] = NOOP

        w_head = self._get_group(welo)
        b_head = self._get_group(belo)

        cd = self.cheatdata[gidx]

        return {
            "input": inp,
            "w_target": w_tgt,
            "b_target": b_tgt,
            "opening": opening,
            "w_head": w_head,
            "b_head": b_head,
            "nhead": len(self.elo_edges),
            "n_inp": n_inp,
            "cheatdata": cd,
        }


def load_data(dirname, load_cheatdata=False):
    md = np.load(os.path.join(dirname, "md.npy"), allow_pickle=True).item()
    with open(f"{dirname}/fmd.json") as f:
        fmd = json.load(f)
    data = {
        "md": md,
        "fmd": fmd,
        "mvids": np.memmap(
            os.path.join(dirname, "mvids.npy"),
            mode="r",
            dtype="int16",
            shape=md["nmoves"],
        ),
        "elos": np.memmap(
            os.path.join(dirname, "whitened_elos.npy"),
            mode="r",
            dtype="float32",
            shape=(fmd["ngames"], 2),
        ),
        "train": np.memmap(
            os.path.join(dirname, "train.npy"),
            mode="r",
            dtype="int64",
            shape=tuple(fmd["train_shape"]),
        ),
        "val": np.memmap(
            os.path.join(dirname, "val.npy"),
            mode="r",
            dtype="int64",
            shape=tuple(fmd["val_shape"]),
        ),
        "test": np.memmap(
            os.path.join(dirname, "test.npy"),
            mode="r",
            dtype="int64",
            shape=tuple(fmd["test_shape"]),
        ),
    }

    if load_cheatdata:
        data["cheatdata"] = np.load(
            os.path.join(dirname, "cheatdata.npy"), allow_pickle=True
        ).item()
    return data


class MMCDataModule(L.LightningDataModule):
    def __init__(
        self,
        datadir,
        elo_edges,
        max_seq_len,
        batch_size,
        num_workers,
        load_cheatdata=False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.elo_edges = elo_edges
        if len(self.elo_edges) == 0 or self.elo_edges[-1] < float("inf"):
            self.elo_edges.append(float("inf"))
        self.load_cheatdata = load_cheatdata
        self.__dict__.update(load_data(datadir, load_cheatdata))
        # min_moves is the minimum game length that can be included in the dataset
        # we subtract one here so that it now represents the minimum number of moves that the
        # model must see before making its first prediction
        self.opening_moves = self.fmd["min_moves"] - 1

    def setup(self, stage):
        if stage == "fit":
            self.trainset = MMCDataset(
                self.max_seq_len,
                self.opening_moves,
                self.train,
                self.mvids,
                self.elos,
                self.elo_edges,
            )
            self.valset = MMCDataset(
                self.max_seq_len,
                self.opening_moves,
                self.val,
                self.mvids,
                self.elos,
                self.elo_edges,
            )
        if stage == "validate":
            self.valset = MMCDataset(
                self.max_seq_len,
                self.opening_moves,
                self.val,
                self.mvids,
                self.elos,
                self.elo_edges,
            )

        if stage in ["test", "predict"]:
            if self.load_cheatdata:
                self.testset = MMCCheatingDataset(
                    self.max_seq_len,
                    self.opening_moves,
                    self.test,
                    self.cheatdata,
                    self.mvids,
                    self.elos,
                    self.elo_edges,
                )
            else:
                self.testset = MMCDataset(
                    self.max_seq_len,
                    self.opening_moves,
                    self.test,
                    self.mvids,
                    self.elos,
                    self.elo_edges,
                )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_worker,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        cfn = cheat_collate_fn if self.load_cheatdata else collate_fn
        return DataLoader(
            self.testset,
            collate_fn=cfn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return self.predict_dataloader()
