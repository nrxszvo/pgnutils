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

from mmc import MimicChessCoreModule, MMCModuleArgs, MimicChessHeadModule
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
    "--name",
    default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="experiment name for log files and checkpoints",
)
parser.add_argument(
    "--train_heads",
    action="store_true",
    help="Train the head for a specific elo range using an existing MMC core checkpoint",
)
parser.add_argument(
    "--ckpt",
    default=None,
    help="MMC checkpoint from which to resume training",
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

    if args.train_heads:
        save_path = os.path.join(args.save_path, "heads", args.name)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "core_ckpt.txt"), "w") as f:
            f.write(args.core_ckpt)
    else:
        save_path = os.path.join(args.save_path, "core", args.name)
        os.makedirs(save_path, exist_ok=True)

    shutil.copyfile(args.cfg, os.path.join(save_path, args.cfg))

    # torch_dist_init()
    devices = int(os.environ.get("WORLD_SIZE", 1))

    n_workers = os.cpu_count() // devices
    os.environ["OMP_NUM_THREADS"] = str(n_workers)

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    model_args = ModelArgs(cfgyml.model_args)
    model_args.n_elo_heads = len(cfgyml.elo_edges) + 1

    def train_model(name, datadir, savepath):
        dm = MMCDataModule(
            datadir,
            cfgyml.elo_edges,
            model_args.max_seq_len,
            cfgyml.batch_size,
            n_workers,
        )
        module_args = MMCModuleArgs(
            name,
            os.path.join(savepath, "ckpt"),
            model_args,
            dm.min_moves,
            NOOP,
            cfgyml.lr_scheduler_params,
            cfgyml.max_steps,
            cfgyml.val_check_steps,
            cfgyml.random_seed,
            cfgyml.strategy,
            devices,
        )

        mmc = MimicChessCoreModule(module_args)

        nweights, nflpweights = mmc.num_params()
        est_tflops = 6 * nflpweights * cfgyml.batch_size * model_args.max_seq_len / 1e12
        print(f"# model params: {nweights:.2e}")
        print(f"estimated TFLOPs: {est_tflops:.1f}")

        mmc.fit(dm, ckpt=args.ckpt)

    datadir = cfgyml.datadir
    train_model(args.name, datadir, save_path)


if __name__ == "__main__":
    main()
