import argparse
from pathlib import Path
import os
import sys
import numpy as np
from pgn.py.lib.reconstruct import reconstruct

import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from config import get_config
from mmc import MimicChessCoreModule, MMCModuleArgs, MimicChessHeadModule
from mmcdataset import MMCDataModule, NOOP
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
    nfail = 0
    ngm = 0
    for tokens, probs, tgt in outputs:
        matches = (tokens == tgt).sum(dim=-1, keepdim=True)
        matches[tgt == NOOP] = 0
        npred = (tgt != NOOP).sum()
        ntotalpred += npred
        ntotalmatch += matches.sum()
        ngm += tokens.shape[0]
        for gm in tokens:
            try:
                reconstruct(gm[:, 0].cpu().numpy())
            except Exception as e:
                nfail += 1

    print(f"{ngm-nfail} out of {ngm} are legal games")
    print(f"Top 3 accuracy: {100*ntotalmatch/ntotalpred:.2f}%")


if __name__ == "__main__":
    main()
