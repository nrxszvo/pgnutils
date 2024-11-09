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
from mmc import MimicChessModule
from mmcdataset import MMCDataModule, NOOP
from model import ModelArgs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default="cfg.yml", help="yaml config file")
parser.add_argument(
    "--save_path",
    default="outputs",
    help="folder for saving config and checkpoints",
)
parser.add_argument(
    "--outfn",
    default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="prediction file name",
)
parser.add_argument(
    "--ngpu", default=1, type=int, help="number of gpus per training trial"
)


def main():
    args = parser.parse_args()
    cfgyml = get_config(args.cfg)
    os.makedirs(args.save_path, exist_ok=True)
    shutil.copyfile(args.cfg, f"{args.save_path}/{args.outfn}.yml")
    model_parallel_size = 1
    if not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.distributed.init_process_group("nccl")
        else:
            torch.distributed.init_process_group("gloo")

    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

    torch.manual_seed(cfgyml.random_seed)

    model_args = ModelArgs(cfgyml.model_args)
    model_args.n_output_heads = len(cfgyml.elo_edges) + 1

    dm = MMCDataModule(
        cfgyml.datadir,
        cfgyml.elo_edges,
        model_args.max_seq_len,
        cfgyml.batch_size,
        os.cpu_count() - 1,
    )
    mmc = MimicChessModule(
        args.outfn,
        os.path.join(args.save_path, "models"),
        model_args,
        dm.min_moves,
        NOOP,
        cfgyml.lr_scheduler_params,
        cfgyml.max_steps,
        cfgyml.val_check_steps,
        cfgyml.random_seed,
        cfgyml.strategy,
        args.ngpu,
    )
    mmc.fit(dm)


if __name__ == "__main__":
    main()
