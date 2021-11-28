from matplotlib import pyplot as plt

import eisplottingtool as ept


def main():
    path = r"C:\Users\ueli\Desktop\Sauter Ulrich\polishing-FC"
    file1 = \
        r"\20201021_Rabeb_LLZTO_Batch4_NOHT_Li300C_3mm_0p7th_EIS_01_PEIS_C16_01_PEIS_C02.mpr"
    file2 = \
        r"\20201126_Rabeb_LLZTO_Batch4_polished_0p55mm_HT400C_Li300C_3mm_0p55th_EIS_01_PEIS_C12.mpr"
    file3 = \
        r"\20201217_Rabeb_LLZTO_Batch4_polished_0p33mm_HT400C_Li300C_3mm_0p33th_FC_C16.mpr"

    # file4 = r"\20210603_B6_water-4weeks-FC_01_PEIS_C03.mpr"

    files = [file1, file2, file3]

    name = [r'Deeply protonated', r'150 $\mu m$ polished',
            r'370 $\mu m$ polished']

    fit_res = [10, 987, 3e-10, 1, 1617.3, 4.5e-8, 0.86, 567, 2]
    fit_circuits = ['R0-p(R1,CPE1)-p(R2,CPE2)-Ws1', 'R0-p(R1,CPE1)-Ws1',
                    'R0-p(R1,CPE1)-p(R2,CPE2)']
    fit_guesses = [[10, 1000, 1e-2, 0.5, 1000, 1e-8, 1, 300, 1e-8],
                   [10, 1000, 1e-2, 0.5, 300, 1e-8],
                   [10, 1000, 1e-2, 0.5, 1000, 1e-8, 1]]
    # fit_bounds = [[(0.0, 100), (0, 2000), (1e-12, 1e-7),
    #                (0, 1), (0, 3200), (1e-7, 1e-3),
    #                (0, 1), (0, 4000), (1e-8, 1000)],
    #               [(0.0, 100), (0, 3000), (1e-11, 1),
    #                (0, 1), (0, 2000), (1e-3, 1000)],
    #               [(0.0, 100), (0, 3000), (1e-11, 1),
    #                (0, 1), (0, 2500), (1e-12, 1e-7),
    #                (0, 1)]]
    indexes = [-1, -1, -1]

    cell_3mm = ept.Cell(3, 0.7)

    fig, axs = ept.create_fig(1, 1)

    for i, file in enumerate(files):
        print(file)
        file_path = path + file
        data = ept.load_data(file_path)

        cycle = data[indexes[i]]

        cycle.plot_nyquist(
                axs,
                plot_range=(-10, 260/0.07),
                label=name[i],
                scale=1
                )
        cycle.fit_nyquist(
                axs,
                fit_circuits[i],
                fit_guess=fit_guesses[i],
                # fit_bounds=fit_bounds[i],
                draw_circle=True
                )
        print("Plotted")
        break
    # now = datetime.now().strftime("%H_%M")
    # ept.save_fig(path + rf"\polishing_{now}.png")
    plt.show()


if __name__ == "__main__":
    print("Start")
    main()
    print("End")
