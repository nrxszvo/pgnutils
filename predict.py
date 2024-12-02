import argparse
import os

import json
from pgn.py.lib.reconstruct import count_invalid

import torch
from config import get_config
from mmc import MimicChessCoreModule, MMCModuleArgs
from mmcdataset import MMCDataModule, NOOP
from model import ModelArgs


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument("--bs", default=None, type=int, help="batch size")


def select_heads(data, heads):
    hdata = torch.index_select(data, 0, heads[:, 0])
    bhdata = torch.index_select(data, 0, heads[:, 1])
    hdata[:, :, 1::2] = bhdata[:, :, 1::2]
    return hdata


def rand_select_heads(data, batch_size):
    n_heads = data.shape[0] // batch_size
    heads = torch.empty(batch_size, dtype=torch.int64)
    for i in range(batch_size):
        heads[i] = i * n_heads + torch.randint(high=n_heads, size=(1,))
    return torch.index_select(data, 0, heads)


class LegalGameStats:
    def __init__(self):
        self.nvalid_games = 0
        self.ntotal_games = 0
        self.nvalid_moves = 0
        self.ntotal_moves = 0

    def eval(self, tokens, opening, tgts):
        self.ntotal_games += tokens.shape[0]
        for game in zip(tokens, opening, tgts):
            pred, opn, tgt = game
            nmoves, nfail = count_invalid(pred[0], opn, tgt[0])
            self.nvalid_moves += nmoves - nfail
            self.ntotal_moves += nmoves
            if nfail == 0:
                self.nvalid_games += 1

    def report(self):
        print(f"Legal game frequency: {100*self.nvalid_games/self.ntotal_games:.1f}%")
        print(f"Legal move frequency: {100*self.nvalid_moves/self.ntotal_moves:.1f}%")


class HeadStats:
    def __init__(self, nheads, batch_size):
        self.nheads = nheads
        self.head_matches = 0
        self.skewed_matches = 0
        self.skewed_total = 0
        self.adj_matches = 0
        self.total_preds = 0
        self.stds = []

    def eval(self, probs, heads, tgts):
        _, seqlen = probs.shape
        head_probs = probs.reshape(-1, self.nheads, seqlen)
        stds = torch.std(head_probs, dim=1)
        self.stds.append(stds.mean())
        max_heads = head_probs.max(dim=1)[1]
        head_matches = max_heads == heads[:, 0:1]
        head_matches[:, 1::2] = max_heads[:, 1::2] == heads[:, 1:2]
        head_matches[tgts[:, 0] == NOOP] = 0
        self.head_matches += head_matches.sum()
        for i, std in enumerate(stds):
            idx = (std > 0.02).nonzero()[:, 0]
            self.skewed_matches += head_matches[i, idx].sum()
            self.skewed_total += idx.shape[0]

        adj_matches = (max_heads == heads[:, 0:1] - 1) | (
            max_heads == heads[:, 0:1] + 1
        )
        adj_matches[:, 1::2] = (max_heads[:, 1::2] == (heads[:, 1:2] - 1)) | (
            max_heads[:, 1::2] == (heads[:, 1:2] + 1)
        )
        adj_matches[tgts[:, 0] == NOOP] = 0
        self.adj_matches += adj_matches.sum()

        npred = (tgts != NOOP).sum()
        self.total_preds += npred

    def report(self):
        print(f"Head accuracy: {100*self.head_matches/self.total_preds:.2f}%")
        print(f"Adjacent Head accuracy: {100*self.adj_matches/self.total_preds:.2f}%")
        print(f"Head std: {100*torch.tensor(self.stds).mean():.2f}%")
        print(f"Skewed accuracy: {100*self.skewed_matches/self.skewed_total:.2f}%")


class MoveStats:
    def __init__(self, elo_edges, ns=[1, 3, 5]):
        self.edges = elo_edges
        self.nheads = len(elo_edges)
        self.stats = [{"n": n, "matches": [0] * self.nheads} for n in ns]
        self.total_preds = [0] * self.nheads

    def eval(self, tokens, heads, tgts):
        for i in range(self.nheads):
            for j in range(heads.shape[1]):
                idx = (heads[:, j] == i).nonzero()[:, 0]
                self.total_preds[i] += (torch.index_select(tgts, 0, idx) != NOOP).sum()

        for s in self.stats:
            move_matches = (tokens[:, : s["n"]] == tgts).sum(dim=1, keepdim=True)
            move_matches[tgts == NOOP] = 0
            for i in range(self.nheads):
                for j in range(heads.shape[1]):
                    idx = (heads[:, j] == i).nonzero()[:, 0]
                    s["matches"][i] += torch.index_select(move_matches, 0, idx).sum()

    def report(self):
        for s in self.stats:
            tpred = 0
            tmatch = 0
            print(f'Top {s["n"]} accuracy:')
            for i in range(self.nheads):
                if self.total_preds[i] > 0:
                    e = self.edges[i]
                    print(f"\t{e}: {100*s['matches'][i]/self.total_preds[i]:.2f}%")
                    tpred += self.total_preds[i]
                    tmatch += s["matches"][i]
            print(f"\tOverall: {100*tmatch/tpred:.2f}%")


@torch.inference_mode()
def evaluate(outputs, elo_edges, n_heads):
    gameStats = LegalGameStats()
    bs = outputs[0]["heads"].shape[0]
    headStats = HeadStats(n_heads, bs)
    moveStats = MoveStats(elo_edges)
    randStats = MoveStats(elo_edges)
    nbatch = len(outputs)
    offset = torch.ones((bs, 1), dtype=torch.int32)
    for i in range(bs):
        offset[i, 0] = i * n_heads

    for i, d in enumerate(outputs):
        print(f"Evaluation {int(100*i/nbatch)}% done", end="\r")
        oheads = d["heads"] - offset
        head_tokens = select_heads(d["sorted_tokens"], d["heads"])
        moveStats.eval(head_tokens, oheads, d["targets"])
        gameStats.eval(head_tokens, d["opening"], d["targets"])

        rand_tokens = rand_select_heads(d["sorted_tokens"], bs)
        randStats.eval(rand_tokens, oheads, d["targets"])

        headStats.eval(d["target_probs"], oheads, d["targets"])
    print()
    gameStats.report()
    print("Target head predictions...")
    moveStats.report()
    print("Random head predictions...")
    randStats.report()
    headStats.report()


def predict(cfgyml, n_heads, cp, fmd, n_workers):
    model_args = ModelArgs(cfgyml.model_args)
    model_args.n_elo_heads = n_heads
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
    mmc = MimicChessCoreModule.load_from_checkpoint(cp, params=module_args)
    dm = MMCDataModule(
        cfgyml.datadir,
        cfgyml.elo_edges,
        model_args.max_seq_len,
        cfgyml.batch_size,
        n_workers,
    )
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

    with open(os.path.join(cfgyml.datadir, "fmd.json")) as f:
        fmd = json.load(f)

    breakpoint()
    n_heads = len(cfgyml.elo_edges) + 1

    outputs = predict(cfgyml, n_heads, args.cp, fmd, n_workers)
    evaluate(outputs, cfgyml.elo_edges, n_heads)


if __name__ == "__main__":
    main()
