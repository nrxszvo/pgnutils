import numpy as np
import matplotlib.pyplot as plt


def game_lengths(npydir):
    md = np.load(f"{npydir}/md.npy", allow_pickle=True).item()
    gs = np.memmap(
        f"{npydir}/gamestarts.npy", mode="r", dtype="int64", shape=md["ngames"]
    )
    gamelengths = np.diff(gs)
    mean = np.mean(gamelengths)
    std = np.var(gamelengths) ** 0.5
    print(
        f"gamelength stats:\n\ttotal games: {len(gamelengths)}\n\tmax length: {gamelengths.max()}\n\tmean: {mean:.2f}, std: {std:.2f}"
    )
    ax = plt.figure().add_subplot()
    ax.hist(gamelengths, bins=100)
    plt.savefig("gamelengths.png", dpi=500)


if __name__ == "__main__":
    import sys

    game_lengths(sys.argv[1])
