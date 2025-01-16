import argparse
import os

import torch

from config import get_config
from mmc import MimicChessModule, MMCModuleArgs
from mmcdataset import NOOP, MMCDataModule
from model import ModelArgs
from utils import AccuracyStats, TargetStats

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
parser.add_argument(
    "--nsamp",
    default=None,
    help="maximum number of samples (games) to process",
    type=int,
)


@torch.inference_mode()
def evaluate(outputs, seq_len, elo_edges):
    nbatch = len(outputs)
    acc_stats = AccuracyStats(seq_len, len(elo_edges), 2 / len(elo_edges))
    tstats = TargetStats()
    for i, d in enumerate(outputs):
        print(f"Evaluation {int(100 * i / nbatch)}% done", end="\r")
        acc_stats.eval(d["sorted_groups"], d["sorted_probs"], d["target_groups"])
        tstats.eval(d["target_probs"], d["adjacent_probs"], d["target_groups"])
    print()
    for line in acc_stats.report() + tstats.report():
        print(line)


def predict(cfgyml, datadir, cp, n_workers, n_samp):
    model_args = ModelArgs(cfgyml.model_args)
    model_args.n_elo_groups = len(cfgyml.elo_edges) + 1
    dm = MMCDataModule(
        datadir=datadir,
        elo_edges=cfgyml.elo_edges,
        max_seq_len=model_args.max_seq_len,
        batch_size=cfgyml.batch_size,
        num_workers=n_workers,
        max_testsamp=n_samp,
    )
    module_args = MMCModuleArgs(
        name=os.path.splitext(os.path.basename(cp))[0],
        outdir=None,
        model_args=model_args,
        opening_moves=dm.opening_moves,
        NOOP=NOOP,
        lr_scheduler_params=cfgyml.lr_scheduler_params,
        max_steps=cfgyml.max_steps,
        val_check_steps=cfgyml.val_check_steps,
        random_seed=cfgyml.random_seed,
        strategy="auto",
        devices=1,
    )
    mmc = MimicChessModule.load_from_checkpoint(cp, params=module_args)
    return mmc.predict(dm), dm.max_seq_len - dm.opening_moves + 1


def main():
    args = parser.parse_args()
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    n_workers = 0  # os.cpu_count() - 1
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    if args.bs:
        cfgyml.batch_size = args.bs

    datadir = cfgyml.datadir if args.datadir is None else args.datadir
    outputs, seq_len = predict(cfgyml, datadir, args.cp, n_workers, args.nsamp)
    report = evaluate(outputs, seq_len, cfgyml.elo_edges)
    if args.report_fn is not None:
        with open(args.report_fn, "w") as f:
            f.write("\n".join(report))


if __name__ == "__main__":
    main()
