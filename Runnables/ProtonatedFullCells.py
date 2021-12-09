import logging
import os
import re

import eisplottingtool as ept
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)


def main():
    path = r"G:\Collaborators\Sauter Ulrich\ProtonatedFullCells"
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

    files = [fc16_deprot_nopres, fc17_prot_pres, fc18_deprot_nopres,
             fc19_prot_nopres, fc19_prot_pres, fc20_prot_pres]
    cell_3mm = ept.Cell(3, 0.7)
    fig, ax = ept.create_fig()

    for i, file in enumerate(files):
        print(file)
        name = re.findall(r'(FC\d*)', file)[0]
        print(name)
        file_path = path + file
        data = ept.load_data(file_path)
        print(f"Number of cycles: {len(data)}")

        for cycle in data:
            cycle.mark_points = []
            cycle.plot_nyquist(
                ax,
                size=8,
                cell=cell_3mm,
                label=name,
                show_freq=True
                )
        break
    plt.show()
    # ept.save_fig(os.path.join(path, "plots", f"protonated_FC.png"))


if __name__ == "__main__":
    logging.info('Started')
    main()
    logging.info('Finished')
