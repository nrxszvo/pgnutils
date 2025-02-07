import argparse
import os
import importlib

import torch

from config import get_config
from utils import get_model_args
import json
from mmcdataset import MMCDataModule
from pgn.py.lib.reconstruct import BoardState

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_dir", default="trained_models/dual_v01")
parser.add_argument("--datadir")


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        return self.model(inp)


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = os.path.join(args.model_dir, "cfg.yml")
    cpfns = list(filter(lambda fn: fn.endswith("ckpt"), os.listdir(args.model_dir)))
    if len(cpfns) > 1:
        print(
            f"Warning: found multiple checkpoints in {args.model_dir}; using {cpfns[0]}"
        )

    cfgyml = get_config(cfg)
    model_args = get_model_args(cfgyml)
    with open(f"{cfgyml.datadir}/fmd.json") as f:
        fmd = json.load(f)
    whiten_params = (fmd["elo_mean"], fmd["elo_std"])

    dm = MMCDataModule(
        datadir=args.datadir,
        elo_edges=cfgyml.elo_params["edges"],
        tc_groups=cfgyml.tc_groups,
        max_seq_len=model_args.max_seq_len,
        batch_size=1,
        num_workers=0,
        whiten_params=whiten_params,
    )
    dm.setup("test")

    modname = args.model_dir.replace("/", ".") + ".model"
    model_mod = importlib.import_module(modname, package=None)
    model = Wrapper(model_mod.Transformer(get_model_args(cfgyml)))
    cp = torch.load(
        os.path.join(args.model_dir, cpfns[0]), map_location=torch.device("cpu")
    )
    model.load_state_dict(cp["state_dict"])
    model.eval()

    for openmvs in [5, 10, 15, 20, 25, 30]:
        print(f"Open moves: {openmvs}")
        for gm in range(10):
            board = BoardState()

            inpdata = dm.testset.__getitem__(gm)
            inp = torch.from_numpy(inpdata["input"][:openmvs]).unsqueeze(0)
            for i in range(openmvs):
                board.update(inp[0, i].item())

            while True:
                mv_pred, elo_pred = model(inp)
                mvids = mv_pred[0, -1, -1].argsort(descending=True)
                best = None
                for mvid in mvids:
                    try:
                        board.update(mvid)
                        best = mvid
                        break
                    except Exception as e:
                        continue

                if best is None:
                    print(
                        f"\tGame {gm} couldnt find a legal move at move {inp.shape[1]}"
                    )
                    break
                if board.board.is_game_over():
                    outcome = board.board.outcome()
                    print(
                        f"\tGame {gm} outcome: {outcome.termination.name} after {inp.shape[1]} moves"
                    )
                    break
                inp = torch.cat([inp, torch.tensor([[best]], dtype=torch.int32)], dim=1)
