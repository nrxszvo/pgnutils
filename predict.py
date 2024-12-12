import argparse
import json
import os

import torch

from config import get_config
from mmc import MimicChessCoreModule, MMCModuleArgs
from mmcdataset import NOOP, MMCDataModule
from model import ModelArgs
from utils import (
    HeadStats,
    LegalGameStats,
    MoveStats,
    CheatStats,
    rand_select_heads,
    select_heads,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument("--bs", default=None, type=int, help="batch size")


@torch.inference_mode()
def evaluate(outputs, elo_edges, n_heads):
    gameStats = LegalGameStats()
    bs = outputs[0]["heads"].shape[0]
    headStats = HeadStats(n_heads, bs)
    moveStats = MoveStats(elo_edges)
    randStats = MoveStats(elo_edges)
    cheatStats = CheatStats(elo_edges)
    nbatch = len(outputs)

    for i, d in enumerate(outputs):
        print(f"Evaluation {int(100*i/nbatch)}% done", end="\r")
        idx = (d["cheatdata"][:, 0] == -1).nonzero()[:, 0]
        head_idx = torch.repeat_interleave(idx * n_heads, n_heads)
        head_idx = head_idx.reshape(-1, n_heads) + torch.arange(n_heads)
        head_idx = head_idx.reshape(-1)
        if len(idx) > 0:
            head_tokens = select_heads(d["sorted_tokens"], d["offset_heads"])[idx]
            moveStats.eval(head_tokens, d["heads"][idx], d["targets"][idx])
            gameStats.eval(head_tokens, d["opening"][idx], d["targets"][idx])
            rand_tokens = rand_select_heads(d["sorted_tokens"], bs)[idx]
            randStats.eval(rand_tokens, d["heads"][idx], d["targets"][idx])

            headStats.eval(
                d["target_probs"][head_idx], d["heads"][idx], d["targets"][idx]
            )

        head_probs = select_heads(d["target_probs"], d["offset_heads"])
        cheatStats.eval(head_probs, d["cheat_probs"], d["cheatdata"], d["heads"])

    print()
    gameStats.report()
    print("Target head predictions...")
    moveStats.report()
    print("Random head predictions...")
    randStats.report()
    headStats.report()
    cheatStats.report()


def predict(cfgyml, n_heads, cp, fmd, n_workers):
    model_args = ModelArgs(cfgyml.model_args)
    model_args.n_elo_heads = n_heads
    module_args = MMCModuleArgs(
        "prediction",
        os.path.join("outputs", "dummy"),
        model_args,
        fmd["min_moves"] - 1,
        NOOP,
        cfgyml.lr_scheduler_params,
        cfgyml.max_steps,
        cfgyml.val_check_steps,
        cfgyml.random_seed,
        "auto",
        1,
        False,
    )
    mmc = MimicChessCoreModule.load_from_checkpoint(cp, params=module_args)
    dm = MMCDataModule(
        cfgyml.datadir,
        cfgyml.elo_edges,
        model_args.max_seq_len,
        cfgyml.batch_size,
        n_workers,
        True,
    )
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

    with open(os.path.join(cfgyml.datadir, "fmd.json")) as f:
        fmd = json.load(f)

    n_heads = len(cfgyml.elo_edges) + 1

    outputs = predict(cfgyml, n_heads, args.cp, fmd, n_workers)
    evaluate(outputs, cfgyml.elo_edges, n_heads)


if __name__ == "__main__":
    main()
