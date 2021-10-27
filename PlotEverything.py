import glob
import os
import re
import Base


def main():
    path = r"Test1.mpr"
    peis_mpr_files = glob.glob(path + r"\*.mpr")
    peis_mpr_files = [path]

    cell_3mm = Base.Cell(3, 0.7)

    print(f"Found {len(peis_mpr_files)} files")

    peis_mpr_files = peis_mpr_files[:3]

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
