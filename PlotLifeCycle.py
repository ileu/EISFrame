import os

import Base


def main():
    path = r"C:\Users\ueli\Desktop\Sauter Ulrich\EIS and cycling raw data"
    file1 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11.mpr"
    file3 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2" \
            r"-10h_C11_again_3_C11.mpr"
    file4 = r"\20200603_LLZTO_polished_ethanol-2h_400C-Ar_Li0p3mm_0p1mApcm2" \
            r"-10h_C13.mpr"
    file5 = r"\ragr_20191204_LLZTO_Ethanol-HT850C_01mApcm2_10h_C03.mpr"

    files = [[file1, file3], file4, file5]

    fig, axs = Base.create_fig(3, 1)
    for i, file in enumerate(files):
        print(file)
        data = Base.load_data(path, file)
        print("Cycles: ", len(data))
        for n, cycle in enumerate(data):
            cycle.plot_lifecycle(
                    axs[i],
                    plot_yrange=(-0.05, 0.05),
                    )
            print(f"cycle {n}")

            print(
                    f"start {cycle['time/s'].iloc[0] / 3600.0}"
                    f", end {cycle['time/s'].iloc[-1] / 3600.0}"
                    )

    Base.save_fig(
            os.path.join(
                    path, "plots", f"test_life.png"

                    )
            )


if __name__ == "__main__":
    print("start")
    main()
    print("end")
