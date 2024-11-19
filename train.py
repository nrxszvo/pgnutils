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

# from mmc import MimicChessCoreModule, MMCModuleArgs, MimicChessHeadModule
from mmcCustom import MimicChessCoreModule, MMCModuleArgs
from mmcdataset import MMCDataModule, NOOP
from model import ModelArgs
import mmcCustom

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
    "--train_heads",
    action="store_true",
    help="Train the head for a specific elo range using an existing MMC core checkpoint",
)
parser.add_argument(
    "--core_ckpt",
    default=None,
    help="core MMC checkpoint to use when training a head output",
)
parser.add_argument("--elo", default=None, help="elo tag for elo-specific head output")


def torch_init():
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.set_float32_matmul_precision("medium")
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
    return model_parallel_size


def main():
    args = parser.parse_args()
    cfgyml = get_config(args.cfg)

    if args.train_heads:
        save_path = os.path.join(args.save_path, "heads", args.outfn)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "core_ckpt.txt"), "w") as f:
            f.write(args.core_ckpt)
    else:
        save_path = os.path.join(args.save_path, "core", args.outfn)
        os.makedirs(save_path, exist_ok=True)

    shutil.copyfile(args.cfg, os.path.join(save_path, args.cfg))

    devices = torch_init()

    torch.manual_seed(cfgyml.random_seed)

    model_args = ModelArgs(cfgyml.model_args)

    def train_model(name, datadir, savepath):
        dm = MMCDataModule(
            datadir,
            model_args.max_seq_len,
            cfgyml.batch_size,
            os.cpu_count() // devices,
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

        if args.train_heads:
            # mmc = MimicChessHeadModule(module_args, args.core_ckpt)
            pass
        else:
            mmc = MimicChessCoreModule(module_args)

        nweights, nflpweights = mmc.num_params()
        est_tflops = 6 * nflpweights * cfgyml.batch_size * model_args.max_seq_len / 1e12
        print(f"# model params: {nweights:.2e}")
        print(f"estimated TFLOPs: {est_tflops:.1f}")

        mmc.fit(dm)

    datadir = cfgyml.datadir
    if args.train_heads:
        for elo in os.listdir(datadir):
            if not os.path.exists(os.path.join(save_path, elo)):
                print(f"training head {elo}")
                name = f"{args.outfn}-{elo}"
                train_model(
                    name, os.path.join(datadir, elo), os.path.join(save_path, elo)
                )
    else:
        train_model(f"{args.outfn}-core", datadir, save_path)


if __name__ == "__main__":
    main()
