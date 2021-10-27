import glob
import os
import re
import Base


def main():
    file1 = r"Test1.mpr"
    file2 = r"Test2.mpr"
    file3 = r"Test3.txt"

    peis_mpr_files = [file1, file2]

    for file in peis_mpr_files:
        image_name = '_'.join(re.split("[_.]", file)[-5:-1])
        print(file, image_name)
        data = Base.load_data(file)
        print("Cycles: ", len(data))
        fig, ax = Base.create_fig()
        for i, cycle in enumerate(data):
            cycle.plot_lifecycle(ax)
        Base.save_fig(
                os.path.join(
                        os.path.dirname(file), "plots", f"{image_name}",
                        f"Life.png"
                )
        )


if __name__ == "__main__":
    print("start")
    main()
    print("end")
