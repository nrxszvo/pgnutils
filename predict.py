import argparse
import os

import json
from pgn.py.lib.reconstruct import count_invalid

import torch
from config import get_config
from mmc import MimicChessCoreModule, MMCModuleArgs, MimicChessHeadModule
from mmcdataset import MMCDataModule, NOOP
from model import ModelArgs


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")


def main():
    args = parser.parse_args()
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    n_workers = os.cpu_count() - 1
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    model_args = ModelArgs(cfgyml.model_args)
    model_args.n_elo_heads = len(cfgyml.elo_edges) + 1

    with open(os.path.join(cfgyml.datadir, "fmd.json")) as f:
        fmd = json.load(f)

    module_args = MMCModuleArgs(
        "prediction",
        os.path.join("outputs", "dummy"),
        model_args,
        fmd["min_moves"] - 1,
        NOOP,
        cfgyml.lr_scheduler_params,
        cfgyml.max_steps,
        cfgyml.val_check_steps,
        cfgyml.random_seed,
        "auto",
        1,
    )
    mmc = MimicChessCoreModule.load_from_checkpoint(args.cp, params=module_args)
    dm = MMCDataModule(
        cfgyml.datadir,
        cfgyml.elo_edges,
        model_args.max_seq_len,
        cfgyml.batch_size,
        n_workers,
    )
    outputs = mmc.predict(dm)

    def select_heads(data, heads):
        hdata = torch.index_select(data, 0, heads[:, 0])
        bhdata = torch.index_select(data, 0, heads[:, 1])
        hdata[:, :, 1::2] = bhdata[:, :, 1::2]
        return hdata

    def evaluate(outputs):
        npass = 0
        ngm = 0
        nvalid = 0
        ntotal = 0
        nbatch = len(outputs)
        top_n_stats = [
            {"n": 3, "move_match": 0},
            {"n": 5, "move_match": 0},
        ]
        ht_matches = 0
        aht_matches = 0
        ntotalpred = 0
        for i, (tokens, probs, heads, opening, tgts) in enumerate(outputs):
            print(f"Evaluation {int(100*i/nbatch)}% done", end="\r")
            _, _, seqlen = probs.shape
            max_heads = (
                probs[:, 0].reshape(-1, model_args.n_elo_heads, seqlen).max(dim=1)[1]
            )
            head_matches = max_heads == heads[:, 0:1]
            head_matches[:, 1::2] = max_heads[:, 1::2] == heads[:, 1:2]
            head_matches[tgts[:, 0] == NOOP] = 0
            ht_matches += head_matches.sum()

            adj_matches = (max_heads == heads[:, 0:1] - 1) | (
                max_heads == heads[:, 0:1] + 1
            )
            adj_matches[:, 1::2] = (max_heads[:, 1::2] == heads[:, 1:2] - 1) | (
                max_heads[:, 1::2] == heads[:, 1:2] + 1
            )
            adj_matches[tgts[:, 0] == NOOP] = 0
            aht_matches += adj_matches.sum()

            npred = (tgts != NOOP).sum()
            ntotalpred += npred

            for s in top_n_stats:
                head_tokens = select_heads(tokens, heads)
                move_matches = (head_tokens[:, : s["n"]] == tgts).sum(
                    dim=1, keepdim=True
                )
                move_matches[tgts == NOOP] = 0
                s["move_match"] += move_matches.sum()

            ngm += head_tokens.shape[0]
            for game in zip(head_tokens, opening, tgts):
                pred, opn, tgt = game
                nmoves, nfail = count_invalid(pred[0], opn, tgt[0])
                nvalid += nmoves - nfail
                ntotal += nmoves
                if nfail == 0:
                    npass += 1
        print()
        print(f"Legal game frequency: {100*npass/ngm:.1f}%")
        print(f"Legal move frequency: {100*nvalid/ntotal:.1f}%")
        print(f"Head accuracy: {100*ht_matches/ntotalpred:.2f}%")
        print(f"Adjacent Head accuracy: {100*aht_matches/ntotalpred:.2f}%")
        for s in top_n_stats:
            print(f"Top {s['n']} accuracy: {100*s['move_match']/ntotalpred:.2f}%")

    evaluate(outputs)


if __name__ == "__main__":
    main()
