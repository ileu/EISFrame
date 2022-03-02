import logging
import os

from matplotlib import pyplot as plt

import eisplottingtool as ept

logging.basicConfig(level=logging.INFO)


class EPTfile:
    def __init__(self, path, name, ignore=False, color=None, marker=None):
        self.ignore = ignore
        self.name = name
        self.path = path
        self.color = color
        self.marker = marker


def main():
    path1 = r"G:\Collaborators\Sauter Ulrich\Projects\ProtonatedFullCells"
    path2 = r"C:\Users\ueli\Desktop\Sauter Ulrich\Projects\ProtonatedFullCells"
    fc16_deprot_nopres = (
        r"\20211126_FC16_B9_water-1w-50C_HT900C-8h_Li"
        r"-4mm_LFP4mm-500um_Cycling_03_PEIS_C05.mpr"
    )
    fc17_prot_pres = (
        r"\20211128_FC17_B9_Water-1w-50C_HT400C-3h_Li-4mm_LFP"
        r"-4mm-500um_Cycling-0p16Nm_06_PEIS_C04.mpr"
    )
    fc18_deprot_nopres = (
        r"\20211201_FC18_Deprotonated-900C-8h_4mmLi_4mmLFP"
        r"-500um_EIS-No pressure-cycling_03_PEIS_C04.mpr"
    )
    fc19_prot_nopres = (
        r"\20211201_FC19_Protonated-400C-3h_4mmLi_4mmLFP"
        r"-500um_cycling_03_PEIS_C05.mpr"
    )
    fc19_prot_pres = (
        r"\20211202_FC19_Protonated-400C-3h_4mmLi_4mmLFP"
        r"-500um_0p16Nm-cycling_03_PEIS_C05.mpr"
    )
    fc20_prot_pres = (
        r"\20211208_FC20_Protonated-400C"
        r"-3h_OneSidePolished_3mmLi_3mmLFP"
        r"-500um_0p16Nm_Cycling_03_PEIS_C04.mpr"
    )
    fc21_prot_nopres = (
        r"\20211209_FC21_protonated-400C"
        r"-3h_onesidepolished_3mmLi_3mmLFP-500um_03_PEIS_C04.mpr"
    )

    fc16 = EPTfile(
        fc16_deprot_nopres,
        "FC16 deprot no press",
        color="blue",
    )
    fc17 = EPTfile(fc17_prot_pres, "FC17 prot with press", color="navy")
    fc18 = EPTfile(fc18_deprot_nopres, "FC18 deprot no press", color="red", marker="s")
    fc19 = EPTfile(fc19_prot_nopres, "FC19 prot no press", color="green")
    fc19_pres = EPTfile(fc19_prot_pres, "FC19 prot with press", color="green")
    fc20 = EPTfile(fc20_prot_pres, "FC20 prot with press", color="blue")
    fc21 = EPTfile(fc21_prot_nopres, "FC21 prot no press", color="orange")

    path = path1
    image_path = r"G:\Collaborators\Sauter Ulrich\Images\FullCell"

    files = [fc16, fc21]

    cell_3mm = ept.Cell(3, 0.7)
    fig, ax = ept.create_fig()
    show_freq = True

    for i, file in enumerate(files):
        if file.ignore:
            continue
        print(file.path)
        file_path = path + file.path
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
                plot_range=(-100, 1000),
                color=file.color,
                marker=file.marker,
            )
            show_freq = False

    ept.save_fig(os.path.join(image_path, f"protonated_FC_good.png"))


if __name__ == "__main__":
    logging.info("Started")
    main()
    logging.info("Finished")
