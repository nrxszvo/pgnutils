import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import lightning as L
import json

NOOP = 2 * 64 + 3  # white's QBISHOP on d1


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

    def __getitem__(self, samp_id):
        if samp_id > self.seq_len:
            breakpoint()
        idx = np.searchsorted(self.indices[:, 0], samp_id, side="right") - 1
        gidx = self.indices[idx, 1]
        offset = self.min_moves + samp_id - idx
        gs = self.gs[gidx]
        ge = gs + offset
        gs = max(gs, ge - self.seq_len - 1)
        inp = np.empty(self.seq_len, dtype="int32")
        n_inp = ge - gs - 1
        inp[: self.seq_len - n_inp] = NOOP
        inp[-n_inp:] = self.mvids[gs : ge - 1]
        welo, belo = self.elo[gidx]
        elo = welo if offset % 2 == 1 else belo
        head = self._get_head(elo)
        return {"input": inp, "target": self.mvids[ge - 1], "head": head}


def load_data(dirname):
    md = np.load(os.path.join(dirname, "md.npy"), allow_pickle=True).item()
    with open(f"{dirname}/fmd.json") as f:
        fmd = json.load(f)
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
        self.num_workers = 0  # num_workers

        self.__dict__.update(load_data(datadir))

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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=lambda id: np.random.seed(id),
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
