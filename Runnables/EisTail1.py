import csv
import glob
import logging
import os

import eisplottingtool as ept
import matplotlib.axes
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def cycles():
    path1 = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch9\20211124_B9P10_Water-1w-60C_HT900C-8h_doubleMelt-2"
    file1 = r"\20211124_B9P10_Water-1w-60C_HT900C-8h_doubleMelt-2_PEIS_01_PEIS_C04.mpr"
    path2 = r"G:\Limit\VMP3 data\Rabeb\Batch4-LLZTO\Acetonitryle-3days"
    file21 = r"\20201204_Rabeb_LLZTO_Batch4_rAcetonitryle" \
                r"-3days_Li300C_3mm_0p7th_PT_C15.mpr"
    file22 = r"\20210210_Rabeb_LLZTO_Batch4_rAcetonitryle" \
                r"-3days_Li300C_3mm_0p7th_PT_After-stop-cell-reassembly_C04" \
                r".mpr"
    path3 = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch9\20211104_B9P4_HT400C-3h_Li-3mm-300C-30min"
    file3 = r"\20211104_B9P4_HT400C-3h_Li-3mm-300C-30min_FCandPT_04_PEIS_C03.mpr"
    path4 = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch9\20211124_B9P10_Water-1w" \
            r"-60C_HT900C-8h_doubleMelt-2"
    file4 = r"\20211126_B9P10_Water-1w-60C_HT900C-8h_doubleMelt" \
            r"-2_FCandPT_03_MB_C04.mpr"
    files = [file4]
    path = path4

    cell_3mm = ept.Cell(3, 0.7)

    for i, file in enumerate(files):
        print(file)
        file_path = path + file
        data = ept.load_data(file_path)

        for n, cycle in enumerate(data):
            fig, ax = ept.create_fig()
            cycle.plot_nyquist(ax)
            cycle.fit_nyquist(
                    ax,
                    'R0-p(R1,CPE1)-Ws1',
                    [0.1, 1694.1, 3.2e-10, 0.9, 620, 1.6],
                    draw_circle=False,
                    path=path + rf"\plots\TailInvestigation\cycle_{i:02d}-{n:03d}_param.txt"
                    )
            fig2, ax2 = ept.create_fig()
            print(f"Current: {np.nanmean(cycle.voltage / cycle.current)}")
            cycle.plot_bode(ax2, param=True)
            plt.show()
            break
            ept.save_fig(
                    os.path.join(
                            path,
                            "plots",
                            "TailInvestigation",
                            f"cycle_{i:02d}-{n:03d}.png"
                            )
                    )


def parameter():
    path1 = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch9\20211124_B9P10_Water-1w" \
           r"-60C_HT900C-8h_doubleMelt-2\plots"
    path2 = r"G:\Limit\VMP3 data\Rabeb\Batch4-LLZTO\Acetonitryle-3days"
    path3 = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch9\20211104_B9P4_HT400C-3h_Li-3mm-300C-30min\plots\TailInvestigation"
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
        ax : matplotlib.axes.Axes
        y = [float(p[param]) for p in params]
        ax.plot(y, 'x')
        ax.set_title(param)
        ax.set_xlabel("Cycles")
        plt.savefig(path3 + rf"\trend_{param}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cycles()
    # parameter()
    logging.info('Finished')

