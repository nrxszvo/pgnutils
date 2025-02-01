import argparse
import tempfile
import os

import torch

from config import get_config
from utils import init_modules
from pgn import prepare_pgn

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pgn")
parser.add_argument("--cfg", default="trained_models/dual-v0.1/cfg.yml")
parser.add_argument(
    "--cp", default="trained_models/dual-v0.1/dual-v0.1-valid_loss=-2.36.ckpt"
)
parser.add_argument("--zst_bin", default="pgn/processZst")
parser.add_argument("--filt_bin", default="pgn/filterData")


@torch.inference_mode()
def evaluate(outputs):
    elo_data = outputs[0][1]
    elo_means = elo_data["elo_mean"][0]
    elo_stds = elo_data["elo_std"][0]
    seqlen = elo_means.shape[0]
    idx = list(range(0, seqlen // 2, seqlen // 6))
    idx += [seqlen]
    idx = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]

    welo_means = [str(int(elo_means[::2][s:e].mean())) for s, e in idx]
    belo_means = [str(int(elo_means[1::2][s:e].mean())) for s, e in idx]

    welo_stds = [str(int(elo_stds[::2][s:e].mean())) for s, e in idx]
    belo_stds = [str(int(elo_stds[1::2][s:e].mean())) for s, e in idx]

    report = [
        "welo means: " + " ".join(welo_means),
        "belo means: " + " ".join(belo_means),
        "welo stds: " + " ".join(welo_stds),
        "belo stds: " + " ".join(belo_stds),
    ]
    return report


def predict(cfgyml, datadir, cp):
    name = (os.path.splitext(os.path.basename(cp))[0],)
    mmc, dm = init_modules(
        cfgyml,
        name,
        "auto",
        1,
        alt_datadir=datadir,
        cp=cp,
        n_workers=0,
    )
    predictions = mmc.predict(dm)
    seqlen = dm.max_seq_len - dm.opening_moves + 1
    return predictions, seqlen


def main():
    args = parser.parse_args()
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    torch.set_float32_matmul_precision("high")

    with tempfile.TemporaryDirectory() as tmpdir:
        prepare_pgn(args.pgn, tmpdir, args.zst_bin, args.filt_bin)
        outputs, seq_len = predict(cfgyml, os.path.join(tmpdir, "filtered"), args.cp)
    report = evaluate(outputs)
    print()
    for line in report:
        print(line)


if __name__ == "__main__":
    main()
