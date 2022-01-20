import glob
import json
import logging
import os
from collections import defaultdict

import eisplottingtool as ept
import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt


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


def cycles():
    path1 = r"G:\Collaborators\Sauter Ulrich\Projects\EIS Tail\Data"
    path2 = r"C:\Users\ueli\Desktop\Sauter Ulrich\Projects\EIS Tail\Data"
    file1 = r"\20201204_Rabeb_LLZTO_Batch4_rAcetonitryle" \
            r"-3days_Li300C_3mm_0p7th_PT_C15.mpr"
    file2 = r"\20210210_Rabeb_LLZTO_Batch4_rAcetonitryle" \
            r"-3days_Li300C_3mm_0p7th_PT_After-stop-cell-reassembly_C04.mpr"
    file3 = r"\20211104_B9P4_HT400C-3h_Li-3mm-300C-30min_FCandPT_02_MB_C03.mpr"
    file4 = r"\20211216_B10P2_Water-1W_HT400C-3h_Li-3mm-280C" \
            r"-30min_150um_FCandPT_03_MB_C13.mpr"
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
    files = [cell4]  # , cell2, cell3, cell4, cell5]

    path = path1
    cell_3mm = ept.Cell(3, 0.7)
    circuit_half = 'R0-p(R1,CPE1)'
    circuit = 'R0-p(R1,CPE1)-Wss1'

    for i, file in enumerate(files[:10]):
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
                        "R3": (res["R1"] + res["R0"], 2*(res["R1"] + res["R0"]))
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
    path1 = r"C:\Users\ueli\Desktop\Sauter Ulrich\Projects\EIS " \
            r"Tail\Data\Acetonitryle"
    path2 = r"G:\Collaborators\Sauter Ulrich\Projects\EIS Tail\Data\Acetonitryle"
    path3 = r"G:\Collaborators\Sauter Ulrich\Projects\EIS " \
            r"Tail\Data\Acetonitryle_restart"

    path = path3

    param_files = glob.glob(path + r"\*param*.txt")
    param_files.sort()
    params = defaultdict(list)
    for n, file in enumerate(param_files):
        print(file)
        with open(file) as f:
            data = json.load(f)
        for d in data:
            params[d['name']].append(d['value'])
    print(params)
    for param in params:
        fig, ax = plt.subplots()
        ax: matplotlib.axes.Axes
        ax.plot(params[param], 'x')
        ax.set_title(param)
        ax.set_xlabel("Cycles")
        plt.tight_layout()
        plt.savefig(path + rf"\trends\trend_{param}")


if __name__ == "__main__":
    logger = logging.getLogger("eisplottingtool")
    logger.setLevel(logging.INFO)

    cycles()
    # parameter()
    print('Finished')
