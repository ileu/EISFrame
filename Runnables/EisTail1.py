import csv
import glob
import os

import eisplottingtool as ept
import pandas as pd
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
    files = [file21, file22]

    cell_3mm = ept.Cell(3, 0.7)

    for i, file in enumerate(files):
        print(file)
        file_path = path2 + file
        data = ept.load_data(file_path)

        for n, cycle in enumerate(data):
            fig, ax = ept.create_fig()
            cycle.plot_nyquist(ax, cell=cell_3mm)
            cycle.fit_nyquist(
                    ax,
                    'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1',
                    [0.1, 1694.1, 3.2e-10, 0.9, 239.33, 2e-6, 0.83, 620, 1.6],
                    cell=cell_3mm,
                    draw_circle=False,
                    path=path2 + rf"\plots\TailInvestigation\cycle_{i}-{n}_param.txt"
                    )
            ept.save_fig(
                    os.path.join(
                            path2,
                            "plots",
                            "TailInvestigation",
                            f"cycle_{i}-{n}.png"
                            )
                    )


def parameter():
    path1 = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch9\20211124_B9P10_Water-1w" \
           r"-60C_HT900C-8h_doubleMelt-2\plots"
    path2 = r"G:\Limit\VMP3 data\Rabeb\Batch4-LLZTO\Acetonitryle-3days"
    param_files = glob.glob(path1 + r"\*param.txt")

    params = []
    for file in param_files:
        with open(file) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            params.append(dict(reader))

    print(params[0].keys())
    fig, ax = plt.subplots()
    y = [float(param['Ws1_R']) for param in params]
    print(y)
    ax.plot(y, 'x')
    plt.show()


if __name__ == "__main__":
    # cycles()
    parameter()
