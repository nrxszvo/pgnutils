import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import lightning as L
import json

NOOP = 2 * 64 + 3  # white's QBISHOP on d1


def init_worker(seed):
    np.random.seed(seed)


class MMCDataset(Dataset):
    def __init__(self, seq_len, min_moves, nsamp, indices, gs, elo, mvids, elo_edges):
        super().__init__()
        self.seq_len = seq_len
        self.min_moves = min_moves
        self.nsamp = nsamp
        self.indices = indices
        self.gs = gs
        self.elo = elo
        self.mvids = mvids
        self.elo_edges = elo_edges

    def __len__(self):
        return self.nsamp

    def _get_head(self, elo):
        for i, upper in enumerate(self.elo_edges):
            if elo < upper:
                return i
        return len(self.elo_edges)

    def __getitem__(self, idx):
        gs, nmoves, gidx = self.indices[idx]
        inp = np.empty(self.seq_len, dtype="int32")
        n_inp = min(self.seq_len, nmoves - 1)
        inp[:n_inp] = self.mvids[gs : gs + n_inp]
        inp[n_inp:] = NOOP

        tgt = np.empty(self.seq_len - self.min_moves, dtype="int64")
        tgt[: n_inp - self.min_moves] = self.mvids[gs + self.min_moves : gs + n_inp]
        tgt[n_inp - self.min_moves :] = NOOP
        # welo, belo = self.elo[gidx]
        # heads = (self._get_head(welo), self._get_head(belo))
        tgt[1::2] = NOOP

        return {
            "input": inp,
            "target": tgt,
            # "heads": heads,
            # "heads": welo.astype("int32"),
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
        "gs": np.memmap(
            os.path.join(dirname, "gs.npy"),
            mode="r",
            dtype="int64",
            shape=fmd["ngames"],
        ),
        "elo": np.memmap(
            os.path.join(dirname, "elo.npy"),
            mode="r",
            dtype="int16",
            shape=(fmd["ngames"], 2),
        ),
        "mvids": np.memmap(
            os.path.join(dirname, "mvids.npy"),
            mode="r",
            dtype="int16",
            shape=md["nmoves"],
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
        self.elo_edges = elo_edges
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.__dict__.update(load_data(datadir))
        self.min_moves = self.fmd["min_moves"]

    def setup(self, stage):
        if stage == "fit":
            self.trainset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.fmd["train_n"],
                self.train,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )
            self.valset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.fmd["val_n"],
                self.val,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )
        if stage == "validate":
            self.valset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.fmd["val_n"],
                self.val,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )

        if stage in ["test", "predict"]:
            self.testset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.fmd["test_n"],
                self.test,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            # sampler=RandomSampler(self.trainset, replacement=True),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return self.predict_dataloader()
