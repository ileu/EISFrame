import os

from src import Base


def main():
    path = r"C:\Users\ragr\Desktop\Follow-up-Camelot\protonation vs deprotonation"
    file1 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11.mpr"
    file2 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11_again_2_C11.mpr"
    file3 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11_again_3_C11.mpr"
    file4 = r"\202000620_LLZTO_polished_Hexane-2min_Ar_CCD_C11.mpr"
    file5 = r"\20201026_Li_3mm_LLZOB4_HT400C_Li3mm_PT_C07.mpr"
    file6 = r"\20210622_B6_P7_MO-7d_HT750C-3h_PT_C01_C13.mpr"
    file7 = r"\20210622_B6_P7_MO-7d_HT750C-3h_PT_again_C13.mpr"
    file8 = r"\20210803_B6_P7_MO-7d_HT750C-3h_PT_2_02_MB_C13.mpr"
    file9 = r"\20210907_B6_P7_MO-7d_HT750C-3h_PT_CORRECT_C13.mpr"
    file10 = r"\20210908_B6_P7_MO-7d_HT750C-3h_PT_3_02_MB_C13.mpr"
    file11 = r"\20210910_B6_P7_MO-7d_HT750C-3h_PT_4_02_MB_C13.mpr"
    file12 = r"\20210913_B6_P7_MO-7d_HT750C-3h_PT_5_02_MB_C04.mpr"
    file13 = r"\20210915_B6_P7_MO-7d_HT750C-3h_PT_6_inversed_02_MB_C04.mpr"
    file14 = r"\20210917_B6_P7_MO-7d_HT750C-3h_PT_7_02_MB_C04.mpr"
    file15= r"\20211004_B8-P2_RT-Water-1w_HT400C_Li-3mm-300C-30min_FCandPT_03_MB_C04.mpr"

    files = [[file1, file2, file3, file4], file15]
    names = ["Hexane", "Deeply protonated"]
    indices_begin = [1, 0]
    indices_end = [-14, None]

    fig, axs = Base.create_fig(2, 1, top_ticks=True)
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
        data = data[indices_begin[i]:indices_end[i]]

        start_time = data[0].time.iloc[0]
        for n, cycle in enumerate(data):
            cycle.time -= start_time
            cycle.plot_lifecycle(
                axs[i],
                plot_yrange=(-0.1, 0.1),
                plot_xrange=(-10,500)
            )
            print(f"cycle {n}")

    Base.save_fig(os.path.join(path, "plots", f"Thermal.png"))


if __name__ == "__main__":
    print("start")
    main()
    print("end")
