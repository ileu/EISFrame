import os

import Base


def main():
    path = r"G:\Collaborators\Sauter Ulrich\Water-param"
    file1 = r"\20200602_LLZTO_polished_Hexane-2min_01_PEIS_C11.mpr"
    file2 = r"\20211004_B8-P2_RT-Water-1w_HT400C_Li-3mm-300C-30min_FCandPT_01_PEIS_C04.mpr"
    file3 = r"\20211006_B8-P2_Water-46C-1w_HT400C_Li-3mm-300C-30min_FCandPT_01_PEIS_C08.mpr"
    file4 = r"\20210603_B6_water-4weeks-FC_01_PEIS_C03.mpr"

    files = [file1, file2, file3, file4]

    name = ["Hexane", "Water 1 week RT", "Water 1 week 50 °C", "Water 4 weeks RT"]

    fit_res = [10, 987, 3e-10, 1, 1617.3, 4.5e-8, 0.86, 567, 2]
    indexes = [-1, -1, -1, -1]

    cell_3mm = Base.Cell(3, 0.7)

    fig, axs = Base.create_fig(4, 1)

    for i, file in enumerate(files):
        print(file)
        file_path = path + file
        data = Base.load_data(file_path)

        cycle = data[indexes[i]]

        cycle.plot_nyquist(
                axs[i],
                cell=cell_3mm,
                plot_range=(-10, 390),
                label=name[i]
                )
        cycle.fit_nyquist(
                axs[i],
                'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1',
                fit_guess=[10, 1000, 1e-2, 0.5, 1000, 1e-8, 1, 300, 1e-12, 1],
                cell=cell_3mm,
                draw_circle=True,
                fit_values=fit_res
                )
        print("Plotted")

    Base.save_fig(path + r"\plot.png")


if __name__ == "__main__":
    print("Start")
    main()
    print("End")
