import argparse
import os
import sys
import torch
from fairscale.nn.model_parallel.initialize import (
    # get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from datetime import datetime
from mmcdataset import MMCDataModule
from mmc import MimicChessModule
from config import get_config
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
    "--ngpu", default=2, type=int, help="number of gpus per training trial"
)


def main():
    args = parser.parse_args()
    cfgyml = get_config(args.cfg)
    # os.makedirs(args.save_path, exist_ok=True)
    # shutil.copyfile(args.cfg, f"{args.save_path}/{args.outfn}.yml")
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

    def get_datamodule(batch_size, npydir):
        return MMCDataModule(
            npydir,
            cfgyml.ntrain,
            cfgyml.nval,
            cfgyml.ntest,
            batch_size,
            os.cpu_count() - 1,
        )

    def get_mmc(
        name,
        outdir,
        model_args,
        lr_scheduler_params,
        batch_size,
        strategy,
        devices,
    ):
        return MimicChessModule(
            name,
            outdir,
            model_args,
            lr_scheduler_params,
            cfgyml.max_steps,
            cfgyml.val_check_steps,
            cfgyml.random_seed,
            strategy,
            devices,
        )

    mmc = get_mmc(
        args.outfn,
        f"{args.save_path}/models",
        ModelArgs(cfgyml.model_args),
        cfgyml.lr_scheduler_params,
        cfgyml.batch_size,
        cfgyml.strategy,
        args.ngpu,
    )
    dm = get_datamodule(cfgyml.batch_size, cfgyml.npydir)
    mmc.fit(dm)


if __name__ == "__main__":
    main()
