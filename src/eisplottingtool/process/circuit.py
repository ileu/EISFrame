import os
import matplotlib.pyplot as plt

from ..utils.UtilFunctions import create_fig, save_fig, plot_legend


def plot_circuit(name, path):
    circuit = "R0-p(R1,CPE1)-Wss1"
    initial_guess = {
        "R0": 91.55829836800659,
        "R1": 1072.8496769984463,
        "CPE1_Q": 1.8784277149199304e-10,
        "CPE1_n": 1.0504983865633575,
        "Wss1_R": 809.2821871862121,
        "Wss1_T": 24.964413479565007,
        "Wss1_n": 0.3484384991224112,
        "Wss2_R": 809.2821871862121,
        "Wss2_T": 24.964413479565007,
        "Wss2_n": 0.3484384991224112,
    }
    print(name)
    data = None

    cycle = data[1]
    fig, ax = create_fig()
    cycle.plot_nyquist(ax=ax, plot_range=(-75, 2200))
    save_fig(os.path.join(path, rf"{name}", f"blank.png"), close=False)
    cycle.plot_fit(
        ax,
        circuit,
        initial_guess,
        color="blue",
        scatter=False,
    )
    save_fig(os.path.join(path, rf"{name}", f"fit.png"), close=False)
    ax.axvline(1973.7, ls="--", label="Total resistance")
    plot_legend(ax)
    save_fig(os.path.join(path, rf"{name}", f"tot.png"), close=False)

    plt.show()
