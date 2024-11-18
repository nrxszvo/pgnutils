import argparse
from pathlib import Path
import os
from pgn.py.lib.reconstruct import count_invalid

import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
)
from config import get_config
from mmc import MimicChessCoreModule, MMCModuleArgs, MimicChessHeadModule
from mmcdataset import MMCDataModule, NOOP
from model import ModelArgs
from train import torch_init

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument(
    "--outfn",
    default=None,
    help="prediction file name",
)


def main():
    args = parser.parse_args()
    cpfn = args.cp
    cfgfn = args.cfg
    outfn = args.outfn
    if outfn is None:
        outfn = cpfn.replace("ckpt", "npy")

    cfgyml = get_config(cfgfn)

    model_parallel_size = torch_init()

    torch.manual_seed(cfgyml.random_seed)

    checkpoints = sorted(Path(args.cp).glob("*.ckpt"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {args.cp}"
    assert (
        model_parallel_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
    ckpt_path = checkpoints[get_model_parallel_rank()]

    model_args = ModelArgs(cfgyml.model_args)

    dm = MMCDataModule(
        cfgyml.datadir,
        model_args.max_seq_len,
        cfgyml.batch_size,
        os.cpu_count() - 1,
    )
    module_args = MMCModuleArgs(
        "test",
        os.path.join("outputs", "core"),
        model_args,
        dm.min_moves,
        NOOP,
        cfgyml.lr_scheduler_params,
        cfgyml.max_steps,
        cfgyml.val_check_steps,
        cfgyml.random_seed,
        cfgyml.strategy,
        1,
    )

    mmc = MimicChessCoreModule.load_from_checkpoint(ckpt_path, params=module_args)
    outputs = mmc.predict(dm)
    ntotalpred = 0
    ntotalmatch = 0
    npass = 0
    ngm = 0
    nvalidmoves = 0
    nbatch = len(outputs)
    for i, (tokens, probs, opening, tgt) in enumerate(outputs):
        print(f"Evaluation {int(100*i/nbatch)}% done", end="\r")
        matches = (tokens == tgt).sum(dim=-1, keepdim=True)
        matches[tgt == NOOP] = 0
        npred = (tgt != NOOP).sum()
        ntotalpred += npred
        ntotalmatch += matches.sum()
        ngm += tokens.shape[0]
        for game in zip(tokens, opening, tgt):
            pred, opn, tgt = game
            nmoves, nfail = count_invalid(pred[:, 0], opn, tgt[:, 0])
            nvalidmoves += nmoves - nfail
            if nfail == 0:
                npass += 1
    print()
    print(f"{npass}/{ngm} are legal games")
    print(f"Average number of legal moves per game: {nvalidmoves/ngm:.1f}")
    print(f"Top 3 accuracy: {100*ntotalmatch/ntotalpred:.2f}%")


if __name__ == "__main__":
    main()
