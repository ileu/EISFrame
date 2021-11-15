import os

import Base


def main():
    path = r"C:\Users\ueli\Desktop\Sauter Ulrich\Solvent Impact"
    file1 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11.mpr"
    file2 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2" \
            r"-10h_C11_again_2_C11.mpr"
    file3 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2" \
            r"-10h_C11_again_3_C11.mpr"
    file4 = r"\20210604_B6_P6_MO-Empa-4w_HT400C_PT_C06.mpr"
    file5 = r"\20210604_B6_water-4weeks-PT_C03.mpr"
    file6 = r"\20210608_B6_P6_EtOH-Empa-4w_HT400C_PT_C02.mpr"

    files = [[file1, file2, file3], file4, file5, file6]
    names = ["Hexane", "Mineral oil 4 weeks", "Water 4 weeks",
             "Ethanol 4 weeks"]

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

    Base.save_fig(os.path.join(path, "plots", f"test_life.png"))


if __name__ == "__main__":
    print("start")
    main()
    print("end")
