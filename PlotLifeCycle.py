import glob
import os
import re
import Base


def main():
    path = r"G:\Collaborators\Sauter Ulrich\Solvent Impact"
    file1 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11.mpr"
    file2 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11_again_2_C11.mpr"
    file3 = r"\202000604_LLZTO_polished_Hexane-2min_Ar_0p1mApcm2-10h_C11_again_3_C11.mpr"
    file4 = r"\20210604_B6_P6_MO-Empa-4w_HT400C_PT_C06.mpr"
    file5 = r"\20210604_B6_water-4weeks-PT_C03.mpr"
    file6 = r"\20210608_B6_P6_EtOH-Empa-4w_HT400C_PT_C02.mpr"

    files = [file1, file2, file3, file4, file5, file6]


    selected_axis = 0
    fig, axs = Base.create_fig(4, 1)
    extra_time = 0
    for i, file in enumerate(files):
        image_name = '_'.join(re.split("[_.]", file)[-5:-1])
        print(file, image_name)
        data = Base.load_data(path + file)
        print("Cycles: ", len(data))
        print(f"Time: {extra_time}")
        start_time = data[0].df['time/s'][0]
        for n, cycle in enumerate(data):
            cycle.mark_points = []
            cycle.df['time/s'] += extra_time - start_time
            cycle.plot_lifecycle(axs[selected_axis], plot_yrange=(-0.05, 0.05))
            # time = cycle['time/s'] / 3600.0 - cycle['time/s'].iloc[0] / 3600.0
            # ax.plot(time, cycle['Ns'], color='blue')
            # ax.plot(time, cycle['I/mA'] * 1e3 / 7.0, color='red')
            print(f"cycle {n}")

            print(
                    f"start {cycle['time/s'].iloc[0] / 3600.0}, end {cycle['time/s'].iloc[-1] / 3600.0}"
                    )
        # lim = .07
        # ax.set_ylim(-lim, lim)
        extra_time = data[-1].df['time/s'].iloc[-1]

        if i < 2:
            continue
        selected_axis += 1
        start_time = 0
        extra_time = 0

    Base.save_fig(
            os.path.join(
                    path, "plots", f"test_life.png"

                    )
            )


if __name__ == "__main__":
    print("start")
    main()
    print("end")
