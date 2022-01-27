import glob
import json
import logging
import os
import re
from collections import defaultdict

import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt

import eisplottingtool as ept


class EPTfile:
    def __init__(
        self,
        path,
        name,
        ignore=False,
        diameter=0,
        color=None,
        thickness=1.0,
        circuit1=None,
        circuit2=None,
        initial_par=None
    ):
        self.ignore = ignore
        self.name = name
        self.path = path
        self.thickness = thickness
        self.color = color
        self.circuit1 = circuit1
        self.circuit2 = circuit2
        self.initial_par = initial_par
        self.diameter = diameter


path1 = r"G:\Collaborators\Sauter Ulrich\Projects\EIS Tail\Data"
path2 = r"C:\Users\ueli\Desktop\Sauter Ulrich\Projects\EIS Tail\Data"
file1 = r"\20201204_Rabeb_LLZTO_Batch4_rAcetonitryle-3days_Li300C_3mm_0p7th_PT_C15" \
        r".mpr"
file2 = r"\20210210_Rabeb_LLZTO_Batch4_rAcetonitryle" \
        r"-3days_Li300C_3mm_0p7th_PT_After-stop-cell-reassembly_C04.mpr"
file3 = r"\20211104_B9P4_HT400C-3h_Li-3mm-300C-30min_FCandPT_02_MB_C03.mpr"
file4 = r"\20220106_B10P4_Water-1w_HT400C-3h_Li-3mm-280C" \
        r"-30min_60um_FCandPT_03_MB_C16.mpr"
file5 = r"\20220105_B9P4_HT400C-3h_Li-3mm-300C-30min_PT_Afterstop_02_MB_C03.mpr"
cell1 = EPTfile(
    file1,
    "Acetonitryle",
    color='C7',
    thickness=0.7,
    diameter=3,
    circuit1="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
    initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
)
cell2 = EPTfile(
    file2,
    "Acetonitryle_restart",
    color='C7',
    thickness=0.7,
    diameter=3,
    circuit1="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
    initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
)
cell3 = EPTfile(
    file3,
    "B9P4_HT400C",
    color='C7',
    thickness=0.7,
    diameter=3,
    circuit1="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
    initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
)
cell4 = EPTfile(
    file4,
    "B10P4_Water_HT400C",
    color='C7',
    thickness=0.7,
    diameter=3,
    circuit1="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
    initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
)
cell5 = EPTfile(
    file5,
    "B9P4_HT400C",
    color='C7',
    thickness=0.7,
    diameter=3,
    circuit1="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
    initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
)
files = [cell1]  # , cell2, cell3, cell4, cell5]

path = path2


def single():
    circuit = 'R0-p(R1,CPE1)-Wss1'
    initial_guess = {
        'R0': 91.55829836800659,
        'R1': 1072.8496769984463,
        'CPE1_Q': 1.8784277149199304e-10,
        'CPE1_n': 1.0504983865633575,
        'Wss1_R': 809.2821871862121,
        'Wss1_T': 24.964413479565007,
        'Wss1_n': 0.3484384991224112,
    }
    for i, file in enumerate(files):
        print(file.name)
        file_path = path + file.path
        data = ept.load_data(file_path)

        cycle = data[1]
        fig, ax = ept.create_fig()
        cycle.plot_nyquist(ax=ax, plot_range=(-75, 2200))
        ept.save_fig(
            os.path.join(
                path,
                rf"{file.name}",
                f"blank.png"
            ),
            close=False
        )
        cycle.plot_fit(
            ax,
            circuit,
            initial_guess,
            color='blue',
            scatter=False,
        )
        ept.save_fig(
            os.path.join(
                path,
                rf"{file.name}",
                f"fit.png"
            ),
            close=False
        )
        ax.axvline(1973.7, ls='--', label='Total resistance')
        ept.utils.plot_legend(ax)
        ept.save_fig(
            os.path.join(
                path,
                rf"{file.name}",
                f"tot.png"
            ),
            close=False
        )

        plt.show()


def cycles():
    circuit_half = 'R0-p(R1,CPE1)'
    circuit = 'R0-p(R1,CPE1)-Wss1'

    for i, file in enumerate(files):
        print(file.name)
        file_path = path + file.path
        data = ept.load_data(file_path)
        initial_guess = {
            'R0': 0.1,
            'R1': 1694.1,
            'CPE1_Q': 3.2e-10,
            'CPE1_n': 0.9,
            'Wss1_R': 700,
            'Wss1_T': 1.6,
            'Wss1_n': 0.5
        }

        for n, cycle in enumerate(data):
            print(f"cycle {n} from {len(data)}")
            fig, ax = ept.create_fig()
            fit_path = rf"\cycle_{i:02d}-{n:03d}_param"
            tot_imp = float(
                np.mean(
                    np.abs(cycle.voltage) / 0.007 * 1e3,
                    where=np.logical_and(
                        cycle.time - cycle.time[0] >= 2500,
                        cycle.time - cycle.time[0] <= 3000
                    )
                )
            )
            print(f"Total imp calc: {tot_imp}")
            if np.isnan(tot_imp):
                continue

            # cycle.df = cycle.df[cycle.df["Ns"] == 1]

            cycle.plot_nyquist(ax, plot_range=(-50, tot_imp * 1.1))
            ax.axvline(tot_imp, ls='--', label='Total resistance')
            __, res = cycle.fit_nyquist(
                ax,
                circuit_half,
                initial_guess,
                fit_bounds={
                    "CPE1_n": (0, 1.2),
                    "Wss1_R": (tot_imp * 0.3, tot_imp),
                    "R1": (100, 4000),
                },
                draw_circle=False,
                path=path + rf"\{file.name}" + fit_path + "_0.txt",
                data_slice=slice(3, 50),
            )
            initial_guess['R3'] = (res["R1"] + res["R0"]) * 1.05
            initial_guess.update(res)
            initial_guess['Wss1_R'] = tot_imp - initial_guess['R3']
            print("fit1 done")
            __, res2 = cycle.fit_nyquist(
                ax,
                'R3-Wss1',
                initial_guess,
                fit_bounds={
                    "CPE1_n": (0, 1.2),
                    "Wss1_R": (tot_imp * 0.3, tot_imp),
                    "Wss1_n": (0, 0.5),
                    "R3": (res["R1"] + res["R0"], 2 * (res["R1"] + res["R0"]))
                },
                fit_constants=["R0", "R1", "CPE1_Q", "CPE1_n"],
                draw_circle=False,
                path=path + rf"\{file.name}" + fit_path + "_1.txt",
                tot_imp=tot_imp,
                data_slice=slice(50, None),
            )
            initial_guess.update(res2)
            cycle.plot_fit(
                ax,
                circuit,
                initial_guess,
                color='blue'
            )

            ept.save_fig(
                os.path.join(
                    path,
                    rf"{file.name}",
                    f"cycle_{i:02d}-{n:03d}.png"
                )
            )
            print("DONE")

        break


def parameter():
    ept.utils.set_plot_params()
    for i, file in enumerate(files):
        print(file.name)
        param_files = glob.glob(path + fr"\{file.name}" + r"\*param*.txt")
        param_files.sort()
        params = defaultdict(list)
        for n, f in enumerate(param_files):
            print(f)
            with open(f) as fl:
                data = json.load(fl)
            cycle_nr = float(re.split(r'\.|_|-', f)[-4])
            for d in data:
                params[d['name']].append((cycle_nr, d['value'], d['unit']))
        print(params)
        for param in params:
            fig, ax = plt.subplots()
            ax: matplotlib.axes.Axes
            x_val = [p[0] for p in params[param]]
            y_val = [p[1] for p in params[param]]
            ax.plot(x_val, y_val, 'x')

            ax.set_xlabel("Cycles")
            label = param
            label = label.replace("Wss", "Ws")
            label = label.replace("R3", "R")
            unit = params[param][0][2]
            unit = unit.replace("Ohm^-1", r"\Omega^{-1}")
            unit = unit.replace("Ohm", r"\Omega")
            if unit == '':
                ax.set_ylabel(f"{label}")
            else:
                ax.set_ylabel(f"{label}/${unit}$")
            plt.tight_layout()
            plt.savefig(path + fr"\{file.name}" + rf"\trends\trend_{param}")
            if param == "Wss1_T":
                d = 0.8 * 1e-11  # cm^2/s tau = l^2 /D, l is diffusion length
                length = [np.sqrt(d * p[1]) * 1e4 for p in params[param]]
                fig, ax = plt.subplots()
                ax: matplotlib.axes.Axes
                ax.plot(x_val, length, 'x')
                ax.set_ylabel("Diffusion length/$\mu m$")
                ax.set_xlabel("Cycles")
                plt.tight_layout()
                plt.savefig(path + fr"\{file.name}" + rf"\trends\diffusion_length")


def life():
    for i, file in enumerate(files):
        print(f"life {file.name}")
        file_path = path + file.path
        data = ept.load_data(file_path)
        fig, ax = ept.create_fig()
        for cycle in data[1:]:
            cycle.time = cycle.time - cycle.time[0]
            cycle.plot_lifecycle(
                ax=ax,
                plot_yrange=(-0.025, 0.025),
                plot_xrange=(-1, 21)
            )
            break
        lifepath = path + rf"\{file.name}\life_short"
        print(lifepath)
        ept.save_fig(lifepath)


if __name__ == "__main__":
    logger = logging.getLogger("eisplottingtool")
    logger.setLevel(logging.INFO)

    single()
    # cycles()
    # parameter()
    # life()
    print('Finished')
