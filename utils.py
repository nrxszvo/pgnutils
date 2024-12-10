from collections import defaultdict
import numpy as np
import torch
from pgn.py.lib.reconstruct import count_invalid
from mmcdataset import NOOP


def select_heads(data, heads):
    hdata = torch.index_select(data, 0, heads[:, 0])
    bhdata = torch.index_select(data, 0, heads[:, 1])
    if hdata.ndim == 3:
        hdata[:, :, 1::2] = bhdata[:, :, 1::2]
    elif hdata.ndim == 2:
        hdata[:, 1::2] = bhdata[:, 1::2]
    else:
        raise Exception
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
                itgts = torch.index_select(tgts, 0, idx)
                self.total_preds[i] += (itgts[:, 0, i::2] != NOOP).sum()

        for s in self.stats:
            move_matches = (tokens[:, : s["n"]] == tgts).sum(dim=1, keepdim=True)
            move_matches[tgts == NOOP] = 0
            for i in range(self.nheads):
                for j in range(heads.shape[1]):
                    idx = (heads[:, j] == i).nonzero()[:, 0]
                    imatches = torch.index_select(move_matches, 0, idx)
                    s["matches"][i] += imatches[:, 0, i::2].sum()

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


class CheatStats:
    def __init__(self):
        self.stats = defaultdict(
            lambda: defaultdict(lambda: {"model": 0, "stockfish": 0})
        )

    def eval(self, head_probs, cheatdata):
        for p, cd in zip(head_probs, cheatdata):
            mvid = cd[0].item()
            gidx = cd[1].item()
            if mvid == -1:
                for ofst in [o.item() for o in cd[2:] if o != -1]:
                    self.stats[gidx][ofst]["model"] = p[ofst].item()
            else:
                ofst = cd[2].item()
                self.stats[gidx][ofst]["stockfish"] = p[ofst].item()

    def report(self):
        ratios = []
        breakpoint()
        for gidx, gstats in self.stats.items():
            for ofst, stats in gstats.items():
                ratios.append(stats["stockfish"] / stats["model"])
        mean = np.mean(ratios)
        print(f"Mean ratio of stockfish to model probability: {mean:.4f}")
