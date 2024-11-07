import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import lightning as L


class MMCDataset(Dataset):
    def __init__(self, nsamp, indices, gs, elo, mvids, elo_edges):
        super().__init__()
        self.indices
        self.elo = elo
        self.mvids = mvids
        self.elo_edge = elo_edges
        self.nsamp = nsamp

    def __len__(self):
        self.nsamp

    def _get_head(self, elo):
        for i, upper in enumerate(self.elo_edges):
            if elo < upper:
                return i
        return len(self.elo_edges)

    def __getitem__(self, samp_id):
        idx = np.searchsorted(self.indices[:, 0], samp_id) - 1
        gedge, gidx = self.indices[idx]
        offset = idx - gedge
        gs = self.gs[gidx]
        ge = gs + offset
        mvids = self.mvids[gs:ge]
        welo, belo = self.elo[gidx]
        elo = welo if offset % 2 == 1 else belo
        head = self._get_head(elo)
        return {"input": mvids[:-1], "target": mvids[-1], "head": head}


def load_npy(npydir):
    def path(name):
        return os.path.join(npydir, f"{name}.npy")

    md = (np.load(path("md.npy"), allow_pickle=True),)
    fmd = (np.load(path("filter_md.npy"), allow_pickle=True),)

    return {
        "md": md,
        "fmd": fmd,
        "gs": np.memmap(path("gs.npy"), mode="r", dtype="int64", shape=md["ngames"]),
        "elo": np.memmap(path("elo.npy"), mode="r", dtype="int16", shape=md["ngames"]),
        "mvids": np.memmap(
            path("mivds.npy"), mode="r", dtype="int16", shape=md["nmoves"]
        ),
        "train": np.memmap(
            path("train.npy"), mode="r", dtype="int64", shape=fmd["train_shape"]
        ),
        "val": np.memmap(
            path("val.npy"), mode="r", dtype="int64", shape=fmd["val_shape"]
        ),
        "test": np.memmap(
            path("test.npy"), mode="r", dtype="int64", shape=fmd["test_shape"]
        ),
    }


class MMCDataModule(L.LightningDataModule):
    def __init__(
        self,
        npydir,
        elo_edges,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.datadir = npydir
        self.elo_edges = elo_edges
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.__dict__.update(load_npy(npydir))

    def setup(self, stage):
        if stage == "fit":
            self.trainset = MMCDataset(
                self.fmd["train_n"],
                self.train,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )
            self.valset = MMCDataset(
                self.fmd["val_n"],
                self.val,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )
        if stage == "validate":
            self.valset = MMCDataset(
                self.fmd["val_n"],
                self.val,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )

        if stage in ["test", "predict"]:
            self.testset = MMCDataset(
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
