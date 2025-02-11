import argparse
import os

import shutil
import sys
from datetime import datetime

import torch
from config import get_config
from fairscale.nn.model_parallel.initialize import (
    # get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from utils import init_modules

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default="cfg.yml", help="yaml config file")
parser.add_argument(
    "--save_path",
    default="outputs",
    help="folder for saving config and checkpoints",
)
parser.add_argument(
    "--name",
    default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="experiment name for log files and checkpoints",
)
parser.add_argument(
    "--ckpt",
    default=None,
    help="MMC checkpoint from which to resume training",
)
parser.add_argument(
    "--commit",
    default=None,
    help="current commit associated with this version of codebase",
)


def torch_dist_init():
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.distributed.init_process_group("nccl")
        else:
            torch.distributed.init_process_group("gloo")

    if not model_parallel_is_initialized():
        initialize_model_parallel(model_parallel_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")


def main():
    args = parser.parse_args()
    cfgyml = get_config(args.cfg)
    cfgyml.commit = args.commit

    save_path = os.path.join(args.save_path, args.name)
    os.makedirs(save_path, exist_ok=True)

    cfgyml.save(os.path.join(save_path, args.cfg))
    return

    # torch_dist_init()
    devices = int(os.environ.get("WORLD_SIZE", 1))

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    mmc, dm = init_modules(
        cfgyml,
        args.name,
        cfgyml.strategy,
        devices,
        outdir=os.path.join(save_path, "ckpt"),
    )

    nweights, nflpweights = mmc.num_params()
    est_tflops = (
        6 * nflpweights * cfgyml.batch_size * cfgyml.model_args["max_seq_len"] / 1e12
    )
    print(f"# model params: {nweights:.2e}")
    print(f"estimated TFLOPs: {est_tflops:.1f}")

    mmc.fit(dm, ckpt=args.ckpt)


if __name__ == "__main__":
    main()
