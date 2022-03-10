import os
import re

import imageio

import eisplottingtool as ept


def main():
    path = r"G:\Limit\VMP3 data\Rabeb\Batch4-LLZTO\Acetonitryle-3days"
    filename1 = (
        r"\20201204_Rabeb_LLZTO_Batch4_rAcetonitryle"
        r"-3days_Li300C_3mm_0p7th_PT_C15.txt"
    )
    filename2 = (
        r"\20210210_Rabeb_LLZTO_Batch4_rAcetonitryle"
        r"-3days_Li300C_3mm_0p7th_PT_After-stop-cell-reassembly_C04"
        r".txt"
    )
    filepath1 = path + filename1
    filepath2 = path + filename2
    data_params = [
        "time/s",
        "Ewe/V",
        "freq/Hz",
        "Re(Z)/Ohm",
        "-Im(Z)/Ohm",
        "cycle number",
    ]
    data = ept.load_data(filepath1, data_param=data_params)
    data_switch = len(data)
    data = data + ept.load_data(filepath2, data_param=data_params)
    image_name = re.split(r"\\", filepath1)[-1][:-4]
    print("Animating")
    cell = ept.Cell(3, 0.7)
    print(f"Cell: {cell}")
    with imageio.get_writer(r"Images\tail_animation_wMark.gif", mode="I") as writer:
        for i, cycle in enumerate(data):
            fig, ax = ept.create_fig()
            fig.suptitle("LLZTO_Batch4_rAcetonitryle-3days")
            label = f"Cycle {i + 1}"
            print(label)
            if i < data_switch:
                label += " before stop"
            else:
                label += " after stop"
            cycle.plot_nyquist(
                ax, plot_range=(-0.5, 300), label=label, scale=1, cell=cell
            )
            img_path = os.path.join(
                os.path.dirname(filepath1), "plots", f"{image_name}", f"cycle_{i}.png"
            )
            ept.save_fig(img_path)
            image = imageio.imread(img_path)
            writer.append_data(image)


if __name__ == "__main__":
    print("start")
    main()
    print("end")
