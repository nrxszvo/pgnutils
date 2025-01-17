import numpy as np
import matplotlib.pyplot as plt
import json


def elo_matrix(
    md, welos, belos, edges=[1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
):
    def get_bins(elos):
        diff_mtx = np.subtract.outer(elos, edges)
        diff_mtx[diff_mtx > 0] = float("-inf")
        idx = diff_mtx.argmax(axis=1)
        return idx

    def get_mtx_idx(welos, belos):
        wbin = get_bins(welos)
        bbin = get_bins(belos)
        mtx = np.zeros((len(edges), len(edges)))
        for m, n in zip(wbin, bbin):
            mtx[m, n] += 1
        return mtx

    maxelo = max(welos.max(), belos.max())
    edges.append(maxelo + 1)
    H, w_edges, b_edges = np.histogram2d(welos, belos, bins=(edges, edges))
    fig, ax = plt.subplots()
    ax.pcolormesh(w_edges, b_edges, np.log10(H.T), cmap="rainbow")
    plt.savefig("elo_matrix.png", dpi=500)
    for row in reversed(np.log10(H)):
        for c in row:
            print(f"{c:.2f}", end="   ")
        print()


def elo_hist(
    md,
    welos,
    belos,
    edges=[0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, float("inf")],
):
    x = np.arange(len(edges) - 1)
    width = 0.33
    mult = 0
    ax = plt.figure().add_subplot()
    breakpoint()
    hw, eb = np.histogram(welos, edges)
    hb, eb = np.histogram(belos, edges)
    for name, data in [("white", hw), ("black", hb)]:
        offset = width * mult
        rects = ax.bar(x + offset, data, width, label=name)
        # ax.bar_label(rects, padding=3)
        mult += 1
    ax.set_ylabel("# games")
    ax.set_yscale("log")
    edge_labels = [str(e) for e in edges[1:]]
    edge_labels[-1] = f">{edge_labels[-2]}"
    ax.set_xticks(x + width, edge_labels)
    ax.legend(loc="upper right")

    plt.savefig("elos.png", dpi=500)


def game_lengths(md, gs):
    gamelengths = np.diff(gs)
    mean = np.mean(gamelengths)
    std = np.var(gamelengths) ** 0.5
    print(
        f"gamelength stats:\n\ttotal games: {len(gamelengths)}\n\tmax length: {gamelengths.max()}\n\tmean: {mean:.2f}, std: {std:.2f}"
    )
    ax = plt.figure().add_subplot()
    ax.hist(gamelengths, bins=100)
    plt.savefig("gamelengths.png", dpi=500)


def load_data(npydir):
    with open(f"{npydir}/fmd.json") as f:
        md = json.load(f)
    gs = np.memmap(f"{npydir}/gs.npy", mode="r", dtype="int64", shape=md["ngames"])
    elos = np.memmap(
        f"{npydir}/elo.npy", mode="r", dtype="int16", shape=(md["ngames"], 2)
    )
    welos = elos[:, 0]
    belos = elos[:, 1]
    return md, gs, welos, belos


if __name__ == "__main__":
    import sys

    md, gs, welos, belos = load_data(sys.argv[1])
    elo_matrix(md, welos, belos)
