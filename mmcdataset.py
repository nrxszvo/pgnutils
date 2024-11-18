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
        tgt = d["target"]
        maxinp = max(maxinp, inp.shape[0])
        maxtgt = max(maxtgt, tgt.shape[0])

    bs = len(batch)
    inputs = torch.full((bs, maxinp), NOOP, dtype=torch.int32)
    targets = torch.full((bs, maxtgt), NOOP, dtype=torch.int64)
    openings = torch.empty((bs, openmoves), dtype=torch.int64)

    for i, d in enumerate(batch):
        inp = d["input"]
        tgt = d["target"]
        ni = inp.shape[0]
        nt = tgt.shape[0]
        inputs[i, :ni] = torch.from_numpy(inp)
        targets[i, :nt] = torch.from_numpy(tgt)
        openings[i] = torch.from_numpy(d["opening"])

    return {"input": inputs, "target": targets, "opening": openings}


class MMCDataset(Dataset):
    def __init__(self, seq_len, min_moves, indices, mvids, train_head=False):
        super().__init__()
        self.seq_len = seq_len
        self.min_moves = min_moves
        self.nsamp = len(indices)
        self.indices = indices
        self.mvids = mvids
        self.train_head = train_head

    def __len__(self):
        return self.nsamp

    def __getitem__(self, idx):
        gs, nmoves, code = self.indices[idx]
        n_inp = min(self.seq_len, nmoves - 1)
        inp = np.empty(n_inp, dtype="int32")
        inp[:] = self.mvids[gs : gs + n_inp]

        opening = np.empty(self.min_moves, dtype="int64")
        opening[:] = self.mvids[gs : gs + self.min_moves]

        tgt = np.empty(n_inp + 1 - self.min_moves, dtype="int64")
        tgt[:] = self.mvids[gs + self.min_moves : gs + n_inp + 1]

        if self.train_head:
            if code == 0:
                tgt[1::2] = NOOP
            elif code == 1:
                tgt[::2] = NOOP

        return {
            "input": inp,
            "target": tgt,
            "opening": opening,
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
        max_seq_len,
        batch_size,
        num_workers,
    ):
        super().__init__()
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
                self.train,
                self.mvids,
            )
            self.valset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.val,
                self.mvids,
            )
        if stage == "validate":
            self.valset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.val,
                self.mvids,
            )

        if stage in ["test", "predict"]:
            self.testset = MMCDataset(
                self.max_seq_len,
                self.fmd["min_moves"],
                self.test,
                self.mvids,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_worker,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.testset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return self.predict_dataloader()
