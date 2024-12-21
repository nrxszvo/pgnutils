import argparse
import json
import os

import torch

from config import get_config
from mmc import MimicChessModule, MMCModuleArgs
from mmcdataset import NOOP, MMCDataModule
from model import ModelArgs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument(
    "--datadir",
    default=None,
    help="alternative data directory to use instead of value in cfg file",
)
parser.add_argument("--bs", default=None, type=int, help="batch size")
parser.add_argument(
    "--report_fn", default=None, help="text file to write summary results"
)


@torch.inference_mode()
def evaluate(outputs, elo_edges):
    nbatch = len(outputs)
    mean_err = 0
    mean_scale = 0
    n_pred = 0
    breakpoint()
    for i, d in enumerate(outputs):
        print(f"Evaluation {int(100*i/nbatch)}% done", end="\r")
        m_pred = d["mean_pred"]
        s_pred = d["scale_pred"]
        welos = d["welos"].unsqueeze(-1)
        belos = d["belos"].unsqueeze(-1)
        mean_err += ((m_pred[:, ::2] - welos) ** 2).sum() + (
            (m_pred[:, 1::2] - belos) ** 2
        ).sum()
        mean_scale += s_pred.sum()
        n_pred += m_pred.numel()
    print()
    print(f"Mean RMS: {(mean_err/n_pred)**0.5:.1f}")
    print(f"Mean scale prediction: {mean_scale/n_pred:.2f}")


def predict(cfgyml, datadir, cp, n_workers):
    model_args = ModelArgs(cfgyml.model_args)
    dm = MMCDataModule(
        datadir=datadir,
        elo_edges=cfgyml.elo_edges,
        max_seq_len=model_args.max_seq_len,
        batch_size=cfgyml.batch_size,
        num_workers=n_workers,
    )
    module_args = MMCModuleArgs(
        name=os.path.splitext(os.path.basename(cp))[0],
        outdir=None,
        model_args=model_args,
        opening_moves=dm.opening_moves,
        NOOP=NOOP,
        whiten_params=dm.whiten_params,
        lr_scheduler_params=cfgyml.lr_scheduler_params,
        max_steps=cfgyml.max_steps,
        val_check_steps=cfgyml.val_check_steps,
        random_seed=cfgyml.random_seed,
        strategy="auto",
        devices=1,
    )
    mmc = MimicChessModule.load_from_checkpoint(cp, params=module_args)
    return mmc.predict(dm)


def main():
    args = parser.parse_args()
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    n_workers = os.cpu_count() - 1
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    if args.bs:
        cfgyml.batch_size = args.bs

    datadir = cfgyml.datadir if args.datadir is None else args.datadir
    outputs = predict(cfgyml, datadir, args.cp, n_workers)
    report = evaluate(outputs, cfgyml.elo_edges)
    if args.report_fn is not None:
        with open(args.report_fn, "w") as f:
            f.write("\n".join(report))


if __name__ == "__main__":
    main()
