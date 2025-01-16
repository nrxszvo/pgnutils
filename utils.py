import numpy as np
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
            f"Legal game frequency: {100 * self.nvalid_games / self.ntotal_games:.3f}%"
        )
        lines.append(
            f"Legal move frequency: {100 * self.nvalid_moves / self.ntotal_moves:.3f}%"
        )
        return lines


class TargetStats:
    def __init__(self):
        self.total_preds = 0
        self.sum_t = 0
        self.sum_adj = 0

    def eval(self, tprobs, adjprobs, tgts):
        tprobs[tgts == NOOP] = 0
        adjprobs[tgts == NOOP] = 0
        self.sum_t += tprobs.sum()
        self.sum_adj += adjprobs.sum()
        self.total_preds += (tgts != NOOP).sum()

    def report(self):
        return [
            f"Mean target probability: {100 * self.sum_t / self.total_preds:.2f}%",
            f"Mean adjacent probability: {100 * self.sum_adj / self.total_preds:.2f}%",
        ]


class AccuracyStats:
    def __init__(self, seq_len, n_groups, min_prob, ns=[1, 3]):
        self.stats = [{"n": n, "matches": np.zeros(seq_len)} for n in ns]
        self.min_prob = min_prob
        self.total_preds = np.zeros(seq_len)
        self.histo = np.zeros((n_groups, n_groups))

    def eval(self, tokens, probs, tgts):
        tokens[probs < self.min_prob] = -1
        for i in range(tgts.shape[0]):
            self.histo[tgts[i, 0], tgts[i, 1]] += 1

        for s in self.stats:
            move_matches = (tokens[:, : s["n"]] == tgts[:, None]).sum(dim=1)
            move_matches[tgts == NOOP] = 0
            s["matches"] += move_matches.sum(dim=0).numpy()
        self.total_preds += (tgts != NOOP).sum(dim=0).numpy()

    def report(self):
        lines = []
        for s in self.stats:
            acc = 100 * s["matches"] / self.total_preds
            lines.append(f"Top {s['n']} accuracy:")
            for i, v in enumerate(self.total_preds):
                if v > 0:
                    acc = 100 * s["matches"][i] / v
                    lines.append(f"\t{i}: {acc:.2f}%")
        lines.append("Elo histogram:")
        for wbin, row in enumerate(reversed(self.histo)):
            rowstr = "  "
            for bbin, n in enumerate(row):
                rowstr += f"{n}".rjust(10)
            lines.append(rowstr)

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
                    f"\t\tstockfish < target: {stats.below[0] / stats.below[1]:.4f} ({stats.below[1]} total moves)"
                )
            if stats.above[1] > 0:
                lines.append(
                    f"\t\tstockfish > target: {stats.above[0] / stats.above[1]:.4f} ({stats.above[1]} total moves)"
                )
        return lines
