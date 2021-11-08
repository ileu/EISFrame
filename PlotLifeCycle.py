import os

import Base


def main():
    path = r"G:\Collaborators\Sauter Ulrich\Water-param"
    file1 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11.mpr"
    file2 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11_again_2_C11.mpr"
    file3 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11_again_3_C11.mpr"
    file4 = r"\20211004_B8-P2_RT-Water-1w_HT400C_Li-3mm-300C-30min_FCandPT_03_MB_C04.mpr"
    file5 = r"\20211006_B8-P2_Water-46C-1w_HT400C_Li-3mm-300C-30min_FCandPT_03_MB_C08.mpr"
    file6 = r"\20210604_B6_water-4weeks-PT_C03.mpr"

    files = [[file1, file2, file3], file4, file5, file6]
    names = ["Hexane", "Water 1 week RT", "Water 1 week 50 °C", "Water 4 weeks RT"]

    fig, axs = Base.create_fig(4, 1, top_ticks=True)
    for i, file in enumerate(files):
        print(file)
        data = Base.load_data(path, file)
        print("Cycles: ", len(data))
        axs[i].text(
            .5,
            .9,
            names[i],
            horizontalalignment='center',
            transform=axs[i].transAxes
            )
        for n, cycle in enumerate(data):
            cycle.plot_lifecycle(
                    axs[i],
                    plot_yrange=(-0.05, 0.05)
                    )
            print(f"cycle {n}")

    Base.save_fig(
            os.path.join(
                    path, "plots", f"test_life.png"

                    )
            )


if __name__ == "__main__":
    print("start")
    main()
    print("end")
