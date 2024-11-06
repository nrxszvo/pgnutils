import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L


class MMCDataset(Dataset):
    def __init__(self, elos, gamestarts, moves):
        super().__init__()
        self.elos = elos
        self.gamestarts = gamestarts
        self.moves = moves

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return {"input": [], "target": []}


def collect_npy(npydir):
    md = np.load(f"{npydir}/md.npy", allow_pickle=True).item()
    elos = np.memmap(
        f"{npydir}/elos.npy", mode="r", dtype="int16", shape=(2, md["ngames"])
    )
    gamestarts = np.memmap(
        f"{npydir}/gamestarts.npy", mode="r", dtype="int64", shape=(1, md["ngames"])
    )
    moves = np.memmap(
        f"{npydir}/moves.npy", mode="r", dtype="int16", shape=(2, md["nmoves"])
    )

    return md, elos, gamestarts, moves


class MMCDataModule(L.LightningDataModule):
    def __init__(
        self,
        npydir,
        ntrain,
        nval,
        ntest,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.datadir = npydir
        self.ntrain = ntrain
        self.nval = nval
        self.ntest = ntest
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.md, self.elos, self.gamestarts, self.moves = collect_npy(self.datadir)

    def setup(self, stage):
        if stage == "fit":
            self.trainset = MMCDataset(self.elos, self.gamestarts, self.moves)
            self.valset = MMCDataset(self.elos, self.gamestarts, self.moves)
        if stage == "validate":
            self.valset = MMCDataset(self.elos, self.gamestarts, self.moves)

        if stage in ["test", "predict"]:
            self.testset = MMCDataset(self.elos, self.gamestarts, self.moves)

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
