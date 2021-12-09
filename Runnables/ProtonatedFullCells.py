import logging
from matplotlib import pyplot as plt

import eisplottingtool as ept

logging.basicConfig(level=logging.INFO)


class EPTfile:
    def __init__(self, path, name, ignore=False):
        self.ignore = ignore
        self.name = name
        self.path = path


def main():
    path = r"G:\Collaborators\Sauter Ulrich\ProtonatedFullCells"
    path2 = r"C:\Users\ueli\Desktop\Sauter Ulrich\ProtonatedFullCells"
    fc16_deprot_nopres = r"\20211126_FC16_B9_water-1w-50C_HT900C-8h_Li" \
                         r"-4mm_LFP4mm-500um_Cycling_03_PEIS_C05.mpr"
    fc17_prot_pres = r"\20211128_FC17_B9_Water-1w-50C_HT400C-3h_Li-4mm_LFP" \
                     r"-4mm-500um_Cycling-0p16Nm_06_PEIS_C04.mpr"
    fc18_deprot_nopres = r"\20211201_FC18_Deprotonated-900C-8h_4mmLi_4mmLFP" \
                         r"-500um_EIS-No pressure-cycling_03_PEIS_C04.mpr"
    fc19_prot_nopres = r"\20211201_FC19_Protonated-400C-3h_4mmLi_4mmLFP" \
                       r"-500um_cycling_03_PEIS_C05.mpr"
    fc19_prot_pres = r"\20211202_FC19_Protonated-400C-3h_4mmLi_4mmLFP" \
                     r"-500um_0p16Nm-cycling_03_PEIS_C05.mpr"
    fc20_prot_pres = r"\20211208_FC20_Protonated-400C" \
                     r"-3h_OneSidePolished_3mmLi_3mmLFP" \
                     r"-500um_0p16Nm_Cycling_03_PEIS_C04.mpr"

    fc16 = EPTfile(fc16_deprot_nopres, "FC16 deprot no press")
    fc17 = EPTfile(fc17_prot_pres, "FC17 prot with press")
    fc18 = EPTfile(fc18_deprot_nopres, "FC18 deprot no press")
    fc19 = EPTfile(fc19_prot_nopres, "FC19 prot no press")
    fc19_pres = EPTfile(fc19_prot_pres, "FC19 prot with press")
    fc20 = EPTfile(fc20_prot_pres, "FC20 prot with press")

    files = [fc16, fc17, fc18, fc19, fc19_pres, fc20]

    cell_3mm = ept.Cell(3, 0.7)
    fig, ax = ept.create_fig()
    show_freq = True

    for i, file in enumerate(files):
        if file.ignore:
            continue
        print(file.path)
        file_path = path2 + file.path
        data = ept.load_data(file_path)
        print(f"Number of cycles: {len(data)}")

        for cycle in data:
            cycle.mark_points = []
            cycle.plot_nyquist(
                    ax,
                    size=8,
                    cell=cell_3mm,
                    label=file.name,
                    show_freq=show_freq,
                    plot_range=(-100, 4000)
                    )
            show_freq = False

    plt.show()
    # ept.save_fig(os.path.join(path, "plots", f"protonated_FC.png"))


if __name__ == "__main__":
    logging.info('Started')
    main()
    logging.info('Finished')
