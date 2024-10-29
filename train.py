import argparse
import os
import shutil
from datetime import datetime
import numpy as np
import torch
from ncmdataset import NCMDataModule
from ncm import NeuralChaosModule
from config import get_config

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

    torch.set_default_dtype(
        torch.float32 if cfgyml.dtype == "float32" else torch.float64
    )

    md_file = cfgyml.datafile
    if os.path.isdir(md_file):
        md_file = f"{md_file}/md.npy"
    step_size = np.load(md_file, allow_pickle=True).item()["ndim"]

    def get_datamodule(batch_size, datafile):
        return NCMDataModule(
            datafile,
            cfgyml.dtype,
            cfgyml.ntrain,
            cfgyml.nval,
            cfgyml.ntest,
            cfgyml.npts,
            cfgyml.input_size,
            cfgyml.H,
            cfgyml.stride,
            cfgyml.spacing,
            batch_size,
            os.cpu_count() - 1,
        )

    def get_ncm(
        name,
        outdir,
        nhits_params,
        lr_scheduler_params,
        batch_size,
        strategy,
        devices,
    ):
        return NeuralChaosModule(
            name,
            outdir,
            cfgyml.H,
            cfgyml.input_size,
            step_size,
            nhits_params,
            lr_scheduler_params,
            cfgyml.loss,
            cfgyml.max_steps,
            cfgyml.val_check_steps,
            cfgyml.random_seed,
            batch_size,
            strategy,
            devices,
        )

    ncm = get_ncm(
        args.outfn,
        f"{args.save_path}/models",
        cfgyml.nhits_params,
        cfgyml.lr_scheduler_params,
        cfgyml.batch_size,
        cfgyml.strategy,
        args.ngpu,
    )
    dm = get_datamodule(cfgyml.batch_size, cfgyml.datafile)
    ncm.fit(dm)


if __name__ == "__main__":
    main()
