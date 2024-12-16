import torch
from pgn.py.lib.reconstruct import count_invalid
from mmcdataset import NOOP


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
        lines = []
        lines.append(
            f"Legal game frequency: {100*self.nvalid_games/self.ntotal_games:.3f}%"
        )
        lines.append(
            f"Legal move frequency: {100*self.nvalid_moves/self.ntotal_moves:.3f}%"
        )
        return lines


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
        lines = []
        for s in self.stats:
            tpred = 0
            tmatch = 0
            lines.append(f'Top {s["n"]} accuracy:')
            for i in range(self.nheads):
                if self.total_preds[i] > 0:
                    e = self.edges[i]
                    lines.append(
                        f"\t{e}: {100*s['matches'][i]/self.total_preds[i]:.2f}%"
                    )
                    tpred += self.total_preds[i]
                    tmatch += s["matches"][i]
            lines.append(f"\tOverall: {100*tmatch/tpred:.2f}%")
        return lines


class CheatStats:
    class EloStats:
        def __init__(self, elo):
            self.elo = elo
            self.below = [0, 0]
            self.above = [0, 0]

    def __init__(self, elo_edges):
        self.stats = [CheatStats.EloStats(elo) for elo in elo_edges]

    def eval(self, head_probs, cheat_probs, cheatdata, heads):
        for tps, cps, cd, hs in zip(head_probs, cheat_probs, cheatdata, heads):
            for i, cp in enumerate(cps):
                offset = cd[i, 0]
                tp = tps[offset]
                head = hs[0] if offset % 2 == 0 else hs[1]
                stats = self.stats[head]
                if cp < tp:
                    stats.below[0] += cp / tp
                    stats.below[1] += 1
                else:
                    stats.above[0] += cp / tp
                    stats.above[1] += 1

    def report(self):
        lines = []
        lines.append("Stockfish / Target mean probability ratios:")
        for stats in self.stats:
            lines.append(f"\tElo < {stats.elo}")
            if stats.below[1] > 0:
                lines.append(
                    f"\t\tstockfish < target: {stats.below[0]/stats.below[1]:.4f} ({stats.below[1]} total moves)"
                )
            if stats.above[1] > 0:
                lines.append(
                    f"\t\tstockfish > target: {stats.above[0]/stats.above[1]:.4f} ({stats.above[1]} total moves)"
                )
        return lines
