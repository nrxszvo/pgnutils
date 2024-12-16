import argparse
import json
import os

import torch

from config import get_config
from mmc import MimicChessModule, MMCModuleArgs
from mmcdataset import NOOP, MMCDataModule
from model import ModelArgs
from utils import (
    LegalGameStats,
    MoveStats,
    CheatStats,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument(
    "--datadir",
    default=None,
    help="alternative data directory to use instead of value in cfg file",
)
parser.add_argument("--bs", default=None, type=int, help="batch size")


@torch.inference_mode()
def evaluate(outputs, elo_edges):
    gameStats = LegalGameStats()
    moveStats = MoveStats(elo_edges)
    cheatStats = CheatStats(elo_edges)
    nbatch = len(outputs)

    for i, d in enumerate(outputs):
        print(f"Evaluation {int(100*i/nbatch)}% done", end="\r")
        idx = (d["cheatdata"][:, 0] == -1).nonzero()[:, 0]
        if len(idx) > 0:
            stokens = d["sorted_tokens"][idx]
            tgts = d["targets"][idx]
            openings = d["openings"][idx]
            moveStats.eval(stokens, tgts)
            gameStats.eval(stokens, openings, tgts)

        cheatStats.eval(d["target_probs"], d["cheat_probs"], d["cheatdata"], d["heads"])

    print()
    gameStats.report()
    print("Target head predictions...")
    moveStats.report()
    print("Random head predictions...")
    cheatStats.report()


def predict(cfgyml, datadir, cp, fmd, n_workers):
    model_args = ModelArgs(cfgyml.model_args)
    dm = MMCDataModule(
        datadir=datadir,
        elo_edges=cfgyml.elo_edges,
        max_seq_len=model_args.max_seq_len,
        batch_size=cfgyml.batch_size,
        num_workers=n_workers,
        load_cheatdata=True,
    )
    module_args = MMCModuleArgs(
        name=os.path.splitext(os.path.basename(cp))[0],
        outdir=None,
        model_args=model_args,
        opening_moves=dm.opening_moves,
        NOOP=NOOP,
        lr_scheduler_params=cfgyml.lr_scheduler_params,
        max_steps=cfgyml.max_steps,
        val_check_steps=cfgyml.val_check_steps,
        random_seed=cfgyml.random_seed,
        strategy="auto",
        devices=1,
    )
    mmc = MimicChessModule.load_from_checkpoint(cp, params=module_args)
    return mmc.predict(dm)


def main():
    args = parser.parse_args()
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    n_workers = os.cpu_count() - 1
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    if args.bs:
        cfgyml.batch_size = args.bs

    datadir = cfgyml.datadir if args.datadir is None else args.datadir
    with open(os.path.join(datadir, "fmd.json")) as f:
        fmd = json.load(f)

    outputs = predict(cfgyml, datadir, args.cp, fmd, n_workers)
    evaluate(outputs, cfgyml.elo_edges)


if __name__ == "__main__":
    main()
