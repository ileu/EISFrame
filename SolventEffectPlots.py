import os
from datetime import datetime

import Base


def main():
    path = r"G:\Collaborators\Sauter Ulrich\polishing-FC"
    file1 = r"\20201021_Rabeb_LLZTO_Batch4_NOHT_Li300C_3mm_0p7th_EIS_01_PEIS_C16_01_PEIS_C02.mpr"
    file2 = r"\20201126_Rabeb_LLZTO_Batch4_polished_0p55mm_HT400C_Li300C_3mm_0p55th_EIS_01_PEIS_C12.mpr"
    file3 = r"\20201217_Rabeb_LLZTO_Batch4_polished_0p33mm_HT400C_Li300C_3mm_0p33th_FC_C16.mpr"
    #file4 = r"\20210603_B6_water-4weeks-FC_01_PEIS_C03.mpr"

    files = [file1, file2, file3]

    name = ["Deeply protonated", "150 $\mu m$ polished", "370 $\mu m$ polished"]

    fit_res = [10, 987, 3e-10, 1, 1617.3, 4.5e-8, 0.86, 567, 2]
    indexes = [-1, -1, -1]

    cell_3mm = Base.Cell(3, 0.7)

    fig, axs = Base.create_fig(3, 1)

    for i, file in enumerate(files):
        print(file)
        file_path = path + file
        data = Base.load_data(file_path)

        cycle = data[indexes[i]]

        cycle.plot_nyquist(
                axs[i],
                cell=cell_3mm,
                plot_range=(-10, 200),
                label=name[i]
                )
        # cycle.fit_nyquist(
        #         axs[i],
        #         'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1',
        #         fit_guess=[10, 1000, 1e-2, 0.5, 1000, 1e-8, 1, 300, 1e-12, 1],
        #         cell=cell_3mm,
        #         draw_circle=True,
        #         fit_values=fit_res
        #         )
        print("Plotted")
    now = datetime.now().strftime("%H_%M")
    Base.save_fig(path + rf"\polishing_{now}.png")


if __name__ == "__main__":
    print("Start")
    main()
    print("End")
