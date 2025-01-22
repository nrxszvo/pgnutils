import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "npydir", help="top-level directory containing either fmd.json or block folders"
)


def elo_matrix(
    welos, belos, edges=[1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
):
    """
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
    """
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
    hw, eb = np.histogram(welos, edges)
    hb, eb = np.histogram(belos, edges)
    for name, data in [("white", hw), ("black", hb)]:
        offset = width * mult
        ax.bar(x + offset, data, width, label=name)
        mult += 1
    ax.set_ylabel("# games")
    ax.set_yscale("log")
    edge_labels = [str(e) for e in edges[1:]]
    edge_labels[-1] = f">{edge_labels[-2]}"
    ax.set_xticks(x + width, edge_labels)
    ax.legend(loc="upper right")

    plt.savefig("elos.png", dpi=500)


def time_hist(blocks, edges=[180, 300, 600, 900, 1800, float("inf")]):
    all_hs = None
    for blk in blocks:
        start_times = blk["clk"][blk["gs"]]
        hs, es = np.histogram(start_times, edges)
        if all_hs is None:
            all_hs = hs
        else:
            all_hs += hs
    for h, e in zip(all_hs, es):
        print(f"{e}:{h}")


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


def load_block_data(blockdirs):
    all_welos = np.array([])
    all_belos = np.array([])
    blocks = []
    for dn in blockdirs:
        print(f"loading {dn}...")
        gs = np.memmap(f"{dn}/gamestarts.npy", mode="r", dtype="int64")
        clk = np.memmap(f"{dn}/clk.npy", mode="r", dtype="int16")
        blocks.append({"gs": gs, "clk": clk})
        welos = np.memmap(f"{dn}/welos.npy", mode="r", dtype="int16")
        belos = np.memmap(f"{dn}/belos.npy", mode="r", dtype="int16")
        all_welos = np.concatenate([all_welos, welos[:]])
        all_belos = np.concatenate([all_belos, belos[:]])
    print("done")
    return all_welos, all_belos, blocks


def load_filtered_data(npydir):
    with open(f"{npydir}/fmd.json") as f:
        fmd = json.load(f)
    elos = np.memmap(
        f"{npydir}/elo.npy", mode="r", dtype="int16", shape=(fmd["ngames"], 2)
    )
    welos = elos[:, 0]
    belos = elos[:, 1]
    return welos, belos


if __name__ == "__main__":
    args = parser.parse_args()
    if "fmd.json" in os.listdir(args.npydir):
        welos, belos = load_filtered_data(args.npydir)
    else:
        blockdirs = [
            os.path.abspath(f"{args.npydir}/{dn}")
            for dn in os.listdir(args.npydir)
            if "block-" in dn
        ]
        welos, belos, blocks = load_block_data(blockdirs)

    # elo_matrix(welos, belos)
    time_hist(blocks)
