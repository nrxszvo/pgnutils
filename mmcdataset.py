import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import json

NOOP = 2 * 64 + 3  # white's QBISHOP on d1


def init_worker(seed):
    np.random.seed(seed)


def collate_fn(batch):
    maxinp = 0
    maxtgt = 0
    openmoves = batch[0]["opening"].shape[0]
    for d in batch:
        inp = d["input"]
        tgt = d["w_target"]
        maxinp = max(maxinp, inp.shape[0])
        maxtgt = max(maxtgt, tgt.shape[0])

    bs = len(batch)
    inputs = torch.full((bs, maxinp), NOOP, dtype=torch.int32)
    openings = torch.empty((bs, openmoves), dtype=torch.int64)
    wtargets = torch.full((bs, maxtgt), NOOP, dtype=torch.int64)
    btargets = torch.full((bs, maxtgt), NOOP, dtype=torch.int64)
    heads = torch.empty((bs, 2), dtype=torch.int32)

    for i, d in enumerate(batch):
        inp = d["input"]
        wtgt = d["w_target"]
        btgt = d["b_target"]
        ni = inp.shape[0]
        nt = wtgt.shape[0]
        inputs[i, :ni] = torch.from_numpy(inp)
        wtargets[i, :nt] = torch.from_numpy(wtgt)
        btargets[i, :nt] = torch.from_numpy(btgt)
        openings[i] = torch.from_numpy(d["opening"])
        heads[i, 0] = i * batch[0]["nhead"] + d["w_head"]
        heads[i, 1] = i * batch[0]["nhead"] + d["b_head"]

    return {
        "input": inputs,
        "w_target": wtargets,
        "b_target": btargets,
        "opening": openings,
        "heads": heads,
    }


class MMCDataset(Dataset):
    def __init__(
        self, seq_len, min_moves, indices, mvids, elos, elo_edges, train_head=False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.min_moves = min_moves
        self.nsamp = len(indices)
        self.indices = indices
        self.mvids = mvids
        self.elos = elos
        self.elo_edges = elo_edges
        self.train_head = train_head

    def __len__(self):
        return self.nsamp

    def _get_head(self, elo):
        for i, edge in enumerate(self.elo_edges):
            if elo <= edge:
                return i

    def __getitem__(self, idx):
        gs, nmoves, gidx = self.indices[idx]
        welo, belo = self.elos[gidx]
        n_inp = min(self.seq_len, nmoves - 1)
        inp = np.empty(n_inp, dtype="int32")
        inp[:] = self.mvids[gs : gs + n_inp]

        opening = np.empty(self.min_moves, dtype="int64")
        opening[:] = self.mvids[gs : gs + self.min_moves]

        w_tgt = np.empty(n_inp + 1 - self.min_moves, dtype="int64")
        b_tgt = np.empty(n_inp + 1 - self.min_moves, dtype="int64")
        w_tgt[:] = self.mvids[gs + self.min_moves : gs + n_inp + 1]
        b_tgt[:] = self.mvids[gs + self.min_moves : gs + n_inp + 1]
        w_tgt[1::2] = NOOP
        b_tgt[::2] = NOOP

        w_head = self._get_head(welo)
        b_head = self._get_head(belo)

        return {
            "input": inp,
            "w_target": w_tgt,
            "b_target": b_tgt,
            "opening": opening,
            "w_head": w_head,
            "b_head": b_head,
            "nhead": len(self.elo_edges),
        }


def load_data(dirname):
    md = np.load(os.path.join(dirname, "md.npy"), allow_pickle=True).item()
    with open(f"{dirname}/fmd.json") as f:
        fmd = json.load(f)
    # min_moves is the minimum game length that can be included in the dataset
    # we subtract one here so that it now represents the minimum number of moves that the
    # model must see before making its first prediction
    fmd["min_moves"] -= 1
    return {
        "md": md,
        "fmd": fmd,
        "mvids": np.memmap(
            os.path.join(dirname, "mvids.npy"),
            mode="r",
            dtype="int16",
            shape=md["nmoves"],
        ),
        "elos": np.memmap(
            os.path.join(dirname, "elo.npy"),
            mode="r",
            dtype="int16",
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


class MMCDataModule(L.LightningDataModule):
    def __init__(
        self,
        datadir,
        elo_edges,
        max_seq_len,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.elo_edges = elo_edges
        if len(self.elo_edges) == 0 or self.elo_edges[-1] < float("inf"):
            self.elo_edges.append(float("inf"))
        self.__dict__.update(load_data(datadir))
        self.min_moves = self.fmd["min_moves"]

    def setup(self, stage):
        if stage == "fit":
            self.trainset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.train,
                self.mvids,
                self.elos,
                self.elo_edges,
            )
            self.valset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.val,
                self.mvids,
                self.elos,
                self.elo_edges,
            )
        if stage == "validate":
            self.valset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.val,
                self.mvids,
                self.elos,
                self.elo_edges,
            )

        if stage in ["test", "predict"]:
            self.testset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
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
        return DataLoader(
            self.testset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return self.predict_dataloader()
