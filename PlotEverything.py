import glob
import os
import re

import Base


def main():
    path = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch8"
    peis_mpr_files = glob.glob(path + r"\*\*PEIS*.mpr")
    
    for file in peis_mpr_files:
        image_name = '_'.join(re.split("[_.]", file)[-5:-1])
        print(file, image_name)
        data = Base.load_data(file)
        print("Cycles: ", len(data))
        for i, cycle in enumerate(data):
            fig, ax = Base.create_fig()
            cycle.plot_nyquist(ax)
            Base.save_fig(os.path.join(os.path.dirname(file), "plots", f"{image_name}", f"cycle_{i}.png"))
        break


if __name__ == "__main__":
    print("start")
    main()
    print("end")
