import glob
import json
import re

from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from ..utils import set_plot_params


def plot_trends(name, path):
    set_plot_params()

    print(name)

    param_files = glob.glob(path + rf"\{name}" + r"\*param*.txt")
    param_files.sort()
    params = defaultdict(list)
    for n, f in enumerate(param_files):
        print(f)
        with open(f) as fl:
            data = json.load(fl)
        cycle_nr = float(re.split(r"[._\-]", f)[-4])
        for d in data:
            params[d["name"]].append((cycle_nr, d["value"], d["unit"]))
    print(params)
    for param in params:
        fig, ax = plt.subplots()
        x_val = [p[0] for p in params[param]]
        y_val = [p[1] for p in params[param]]
        ax.plot(x_val, y_val, "x")

        ax.set_xlabel("Cycles")
        label = param
        label = label.replace("Wss", "Ws")
        label = label.replace("R3", "R")
        unit = params[param][0][2]
        unit = unit.replace("Ohm^-1", r"\Omega^{-1}")
        unit = unit.replace("Ohm", r"\Omega")
        if unit == "":
            ax.set_ylabel(f"{label}")
        else:
            ax.set_ylabel(f"{label}/${unit}$")
        plt.tight_layout()
        plt.savefig(path + rf"\{name}" + rf"\trends\trend_{param}")
        if param == "Wss1_T":
            d = 0.8 * 1e-11  # cm^2/s tau = l^2 /D, l is diffusion length
            length = [np.sqrt(d * p[1]) * 1e4 for p in params[param]]
            fig, ax = plt.subplots()
            ax.plot(x_val, length, "x")
            ax.set_ylabel(r"Diffusion length/$\mu m$")
            ax.set_xlabel("Cycles")
            plt.tight_layout()
            plt.savefig(path + rf"\{name}" + rf"\trends\diffusion_length")
