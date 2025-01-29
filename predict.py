import argparse

import torch
import numpy as np

from config import get_config
from utils import AccuracyStats, TargetStats, init_modules

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
    acc_stats = AccuracyStats(seq_len, elo_edges, 2 / len(elo_edges))
    tstats = TargetStats()
    nll = np.array([op["nll"] for op in outputs]).mean()
    for i, d in enumerate(outputs):
        print(f"Evaluation {int(100 * i / nbatch)}% done", end="\r")
        acc_stats.eval(d["sorted_groups"], d["sorted_probs"], d["target_groups"])
        tstats.eval(d["target_probs"], d["adjacent_probs"], d["target_groups"])
    report = acc_stats.report() + tstats.report() + [f"NLL: {nll:.2f}"]
    return report


def predict(cfgyml, datadir, cp, n_samp):
    mmc, dm = init_modules(cfgyml, "auto", 1, alt_datadir=datadir, n_samp=n_samp, cp=cp)
    return mmc.predict(dm), dm.max_seq_len - dm.opening_moves + 1


def main():
    args = parser.parse_args()
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    if args.bs:
        cfgyml.batch_size = args.bs

    datadir = cfgyml.datadir if args.datadir is None else args.datadir
    outputs, seq_len = predict(cfgyml, datadir, args.cp, args.nsamp)
    report = evaluate(outputs, seq_len, cfgyml.elo_edges)
    print()
    for line in report:
        print(line)
    if args.report_fn is not None:
        with open(args.report_fn, "w") as f:
            f.write("\n".join(report))


if __name__ == "__main__":
    main()
