import glob
import os
import re
import Base


def main():
    path = r"G:\Limit\VMP3 data\Rabeb\Ampcera-LLZTO-Batch6\B6-P7_MO-7d_750C-3h"
    peis_mpr_files = glob.glob(path + r"\*.mpr")

    cell_3mm = Base.Cell(3, 0.7)

    for file in peis_mpr_files:
        image_name = '_'.join(re.split("[_.]", file)[-5:-1])
        print(file, image_name)
        data = Base.load_data(file)
        print("Cycles: ", len(data))
        for i, cycle in enumerate(data):
            fig, ax = Base.create_fig()
            cycle.plot_nyquist(ax, cell=cell_3mm)
            Base.save_fig(
                os.path.join(
                    os.path.dirname(file), "plots", f"{image_name}",
                    f"cycle_{i}.png"
                    )
                )


if __name__ == "__main__":
    print("start")
    main()
    print("end")
