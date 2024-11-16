import argparse
from pathlib import Path
import os
import sys
import numpy as np
from pgn.py.lib import reconstruct

import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from config import get_config
from mmcdataset import MMCDataModule, NOOP
from mmc import MimicChessModule
from model import ModelArgs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument(
    "--outfn",
    default=None,
    help="prediction file name",
)


def main():
    torch.set_float32_matmul_precision("high")
    args = parser.parse_args()
    cpfn = args.cp
    cfgfn = args.cfg
    outfn = args.outfn
    if outfn is None:
        outfn = cpfn.replace("ckpt", "npy")

    cfgyml = get_config(cfgfn)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    torch.manual_seed(cfgyml.random_seed)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    checkpoints = sorted(Path(args.cp).glob("*.ckpt"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {args.cp}"
    assert (
        model_parallel_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
    ckpt_path = checkpoints[get_model_parallel_rank()]

    model_args = ModelArgs(cfgyml.model_args)
    model_args.n_output_heads = len(cfgyml.elo_edges) + 1

    dm = MMCDataModule(
        cfgyml.datadir,
        cfgyml.elo_edges,
        model_args.max_seq_len,
        cfgyml.batch_size,
        os.cpu_count() - 1,
    )
    mmc = MimicChessModule.load_from_checkpoint(
        ckpt_path,
        name=args.outfn,
        outdir="outputs/models",
        model_args=model_args,
        min_moves=dm.min_moves,
        NOOP=NOOP,
        lr_scheduler_params=cfgyml.lr_scheduler_params,
    )
    outputs = mmc.predict(dm)
    ntotalpred = 0
    ntotalmatch = 0
    for tokens, tgt in outputs:
        matches = (tokens == tgt).sum(dim=-1, keepdim=True)
        matches[tgt == NOOP] = 0
        npred = (tgt != NOOP).sum()
        ntotalpred += npred
        ntotalmatch += matches.sum()

    print(f"Top 3 accuracy: {100*ntotalmatch/ntotalpred:.2f}%")


if __name__ == "__main__":
    main()
