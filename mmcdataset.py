import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import lightning as L


class MMCDataset(Dataset):
    def __init__(self, indices, gs, elo, mvids, elo_edges):
        super().__init__()
        self.indices = indices
        self.gs = gs
        self.elo = elo
        self.mvids = mvids
        self.elo_edge = elo_edges

    def __len__(self):
        return self.indices.shape[0]

    def _get_head(self, elo):
        for i, upper in enumerate(self.elo_edges):
            if elo < upper:
                return upper
        return len(self.elo_edges)

    def __getitem__(self, idx):
        gsidx, offset = self.indices[idx]
        gs = self.gs[gsidx]
        ge = gs + offset
        mvids = self.mvids[gs:ge]
        welo, belo = self.elo[idx]
        elo = welo if offset % 2 == 1 else belo
        head = self._get_head(elo)
        return {"input": mvids[:-1], "target": mvids[-1], "head": head}


def load_splits(npydir):
    ret = []
    for name in ("gs", "elo", "train", "test", "val"):
        ret.append(np.load(os.path.join(npydir, f"{name}.npy"), allow_pickle=True))
    return ret


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

        self.gs, self.elo, self.trainidx, self.validx, self.testidx = load_splits(
            npydir
        )
        md = np.load(f"{npydir}/md.npy", allow_pickle=True).item()
        self.mvids = np.memmap(
            f"{npydir}/mvids.npy", mode="r", dtype="int16", shape=md["nmoves"]
        )

    def setup(self, stage):
        if stage == "fit":
            self.trainset = MMCDataset(
                self.trainidx, self.gs, self.elo, self.mvids, self.elo_edges
            )
            self.valset = MMCDataset(
                self.validx,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )
        if stage == "validate":
            self.valset = MMCDataset(
                self.validx,
                self.gs,
                self.elo,
                self.mvids,
                self.elo_edges,
            )

        if stage in ["test", "predict"]:
            self.testset = MMCDataset(
                self.testidx,
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
