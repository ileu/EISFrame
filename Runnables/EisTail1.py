import csv
import glob
import logging
import os

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
            circuit=None,
            initial_par=None
    ):
        self.ignore = ignore
        self.name = name
        self.path = path
        self.thickness = thickness
        self.color = color
        self.circuit = circuit
        self.initial_par = initial_par
        self.diameter = diameter


def cycles():
    path = r"G:\Collaborators\Sauter Ulrich\Projects\EIS Tail\Data"
    file1 = r"\20201204_Rabeb_LLZTO_Batch4_rAcetonitryle" \
            r"-3days_Li300C_3mm_0p7th_PT_C15.mpr"
    file2 = r"\20210210_Rabeb_LLZTO_Batch4_rAcetonitryle" \
            r"-3days_Li300C_3mm_0p7th_PT_After-stop-cell-reassembly_C04.mpr"
    file3 = r"\20211104_B9P4_HT400C-3h_Li-3mm-300C-30min_FCandPT_02_MB_C03.mpr"
    file4 = r"\20211124_B9P10_Water-1w-60C_HT900C-8h_doubleMelt" \
            r"-2_FCandPT_02_MB_C04.mpr"
    file5 = r"\20211124_B9P10_Water-1w-60C_HT900C-8h_doubleMelt" \
            r"-2_FCandPT_03_MB_C04.mpr"
    cell1 = EPTfile(
            file1,
            "Acetonitryle",
            color='C7',
            thickness=0.7,
            diameter=3,
            circuit="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
            initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
    )
    cell2 = EPTfile(
            file2,
            "Acetonitryle_restart",
            color='C7',
            thickness=0.7,
            diameter=3,
            circuit="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
            initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
    )
    cell3 = EPTfile(
            file3,
            "B9P4_HT400C",
            color='C7',
            thickness=0.7,
            diameter=3,
            circuit="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
            initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
    )
    cell4 = EPTfile(
            file4,
            "B9P10_double-melt-2_FC",
            color='C7',
            thickness=0.7,
            diameter=3,
            circuit="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
            initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
    )
    cell5 = EPTfile(
            file5,
            "B9P10_double-melt-2_PT",
            color='C7',
            thickness=0.7,
            diameter=3,
            circuit="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
            initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
    )
    files = [cell1, cell2, cell3, cell4, cell5]

    cell_3mm = ept.Cell(3, 0.7)
    circuit = 'R0-p(R1,CPE1)-Ws1'

    for i, file in enumerate(files):
        print(file.name)
        file_path = path + file.path
        data = ept.load_data(file_path)

        for n, cycle in enumerate(data[:10]):
            fig, ax = ept.create_fig()
            cycle.plot_nyquist(ax)
            fit_path = rf"\cycle_{i:02d}-{n:03d}_param.txt"
            params, __ = cycle.fit_nyquist(
                    ax,
                    circuit,
                    [0.1, 1694.1, 3.2e-10, 0.9, 620, 1.6],
                    fit_bounds={"CPE1_n": (0, 2)},
                    draw_circle=False,
                    path=path + rf"\{file.name}" + fit_path
            )
            # fig2, ax2 = ept.create_fig()
            print(f"Total imp calc: {np.nanmean(cycle.voltage[:10] / 0.007 * 1e3)}")
            # cycle.plot_bode(ax2, param_values=params, param_circuit=circuit)

            ept.save_fig(
                    os.path.join(
                            path,
                            rf"{file.name}",
                            f"cycle_{i:02d}-{n:03d}.png"
                    )
            )
            print("DONE")


def parameter():
    path1 = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch9\20211124_B9P10_Water-1w" \
            r"-60C_HT900C-8h_doubleMelt-2\plots"
    path2 = r"G:\Limit\VMP3 data\Rabeb\Batch4-LLZTO\Acetonitryle-3days"
    path3 = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch9\20211104_B9P4_HT400C-3h_Li" \
            r"-3mm-300C-30min\plots\TailInvestigation"
    param_files = glob.glob(path3 + r"\*param.txt")
    param_files.sort()
    params = []
    for file in param_files:
        print(file)
        with open(file) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            params.append(dict(reader))

    print(params)
    for param in params[0]:
        fig, ax = plt.subplots()
        ax: matplotlib.axes.Axes
        y = [float(p[param]) for p in params]
        ax.plot(y, 'x')
        ax.set_title(param)
        ax.set_xlabel("Cycles")
        plt.savefig(path3 + rf"\trend_{param}")


if __name__ == "__main__":
    cycles()
    # parameter()
    logging.info('Finished')
