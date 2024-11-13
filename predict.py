import argparse
import os

import numpy as np
import torch

from config import get_config
from mmcdataset import MMCDataModule
from mmc import MimicChessModule 


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

    mmc.predict(dm)


if __name__ == "__main__":
    main()
