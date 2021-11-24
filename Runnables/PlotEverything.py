import glob
import os
import re

import eisplottingtool as ept


def main():
    path = r"G:\Collaborators\Sauter Ulrich\Solvent Impact"
    peis_mpr_files = glob.glob(path + r"\*EIS*.mpr")

    cell_3mm = ept.Cell(3, 0.7)

    print(f"Found {len(peis_mpr_files)} files")

    for file in peis_mpr_files:
        image_name = '_'.join(re.split("[_.]", file)[-5:-1])
        print(file, image_name)
        data = ept.load_data(file)
        print("Cycles: ", len(data))
        for i, cycle in enumerate(data):
            fig, ax = ept.create_fig()
            cycle.plot_nyquist(ax, cell=cell_3mm)
            cycle.fit_nyquist(
                    ax,
                    'R0-p(R1,CPE1)-p(R2,CPE2)',
                    [10, 500, 1e-8, 0.9, 200, 1e-8, 0.5],
                    cell=cell_3mm,
                    draw_circle=True,
                    )
            ept.save_fig(
                    os.path.join(
                            os.path.dirname(file),
                            "plots",
                            f"{image_name}",
                            f"cycle_{i}.png"
                            )
                    )
        break


if __name__ == "__main__":
    print("start")
    main()
    print("end")
