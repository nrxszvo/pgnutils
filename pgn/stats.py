import numpy as np
import matplotlib.pyplot as plt
import json


def elo_hist(
    elos,
    edges=[0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, float("inf")],
):
    x = np.arange(len(edges) - 1)
    width = 0.33
    mult = 0
    ax = plt.figure().add_subplot()
    hw, eb = np.histogram(elos[:, 0], edges)
    hb, eb = np.histogram(elos[:, 1], edges)
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


def game_lengths(gs):
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
        fmd = json.load(f)
    gs = np.memmap(f"{npydir}/gs.npy", mode="r", dtype="int64", shape=fmd["ngames"])
    elos = np.memmap(
        f"{npydir}/elo.npy", mode="r", dtype="int16", shape=(fmd["ngames"], 2)
    )
    return fmd, gs, elos


if __name__ == "__main__":
    import sys

    fmd, gs, elos = load_data(sys.argv[1])
    elo_hist(elos)
