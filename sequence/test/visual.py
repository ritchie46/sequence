from sequence import utils
import numpy as np


def plot_annealings():
    pct = np.linspace(0, 1)
    plt.plot(pct, utils.annealing_cosine(0, 1, pct))
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_annealings()
