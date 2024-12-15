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
        print(f"Legal game frequency: {100*self.nvalid_games/self.ntotal_games:.1f}%")
        print(f"Legal move frequency: {100*self.nvalid_moves/self.ntotal_moves:.1f}%")


class MoveStats:
    def __init__(self, ns=[1, 3, 5]):
        self.stats = [{"n": n, "matches": 0} for n in ns]
        self.total_preds = 0

    def eval(self, tokens, tgts):
        for s in self.stats:
            move_matches = (tokens[:, : s["n"]] == tgts).sum(dim=1, keepdim=True)
            move_matches[tgts == NOOP] = 0
            s["matches"] += move_matches.sum()
        self.total_preds += (tgts != NOOP).sum()

    def report(self):
        for s in self.stats:
            print(f'Top {s["n"]} accuracy:')
            print(f"\t{100*s['matches']/self.total_preds:.2f}%")


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
        print("Stockfish / Target mean probability ratios:")
        for stats in self.stats:
            print(f"\tElo < {stats.elo}")
            if stats.below[1] > 0:
                print(
                    f"\t\tstockfish < target: {stats.below[0]/stats.below[1]:.4f} ({stats.below[1]} total moves)"
                )
            if stats.above[1] > 0:
                print(
                    f"\t\tstockfish > target: {stats.above[0]/stats.above[1]:.4f} ({stats.above[1]} total moves)"
                )
