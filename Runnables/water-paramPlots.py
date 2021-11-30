import eisplottingtool as ept


def main():
    path = r"G:\Collaborators\Sauter Ulrich\Water-param"
    file1 = r"\20200602_LLZTO_polished_Hexane-2min_01_PEIS_C11.mpr"
    file2 = r"\20211004_B8-P2_RT-Water-1w_HT400C_Li-3mm-300C-30min_FCandPT_01_PEIS_C04.mpr"
    file3 = r"\20211006_B8-P2_Water-46C-1w_HT400C_Li-3mm-300C-30min_FCandPT_01_PEIS_C08.mpr"
    file4 = r"\20210603_B6_water-4weeks-FC_01_PEIS_C03.mpr"

    files = [file1, file2, file3, file4]

    indexes = [-1, -1, -1, -1]

    cell_3mm = ept.Cell(3, 0.7)

    fig, axs = ept.create_fig(4, 1)

    for i, file in enumerate(files):
        print(file)
        file_path = path + file
        data = ept.load_data(file_path)

        cycle = data[indexes[i]]

        cycle.plot_nyquist(axs[i], cell=cell_3mm, plot_range=(-20, 300))
        print("Plotted")

    ept.save_fig(path + r"\plot1.png")


if __name__ == "__main__":
    print("Start")
    main()
    print("End")
