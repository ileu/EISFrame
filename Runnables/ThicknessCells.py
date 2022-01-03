import logging
import sys

from matplotlib import pyplot as plt

import eisplottingtool as ept

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class EPTfile:
    def __init__(
        self,
        path,
        name,
        ignore=False,
        color=None,
        thickness=1.0,
        circuit=None,
        initial_par=None
    ):
        self.ignore = ignore
        self.name = name
        self.path = path
        self.thickness = thickness
        self.color = color
        self.circuit = circuit
        self.initial_par = initial_par


def main():
    # TODO: only normalize to area
    path1 = r"G:\Collaborators\Sauter Ulrich\Projects\Thickness"
    path2 = r"C:\Users\ueli\Desktop\Sauter Ulrich\Projects\Thickness"
    path = path1
    file1 = r"\20211217_B10P2_Water-1W_HT400C-3h_Li-3mm-280C" \
            r"-30min_0um_EIS_01_PEIS_C15.mpr"
    file2 = r"\20211216_B10P2_Water-1W_HT400C-3h_Li-3mm-280C" \
            r"-30min_100um_EIS_01_PEIS_C04.mpr"
    file3 = r"\20211216_B10P2_Water-1W_HT400C-3h_Li-3mm-280C" \
            r"-30min_150um_EIS_01_PEIS_C13.mpr"

    cell0um = EPTfile(
        file1,
        "Protonated ref",
        color='C7',
        thickness=0.7,
        circuit="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
        initial_par=[1, 1500, 1e-8, 0.9, 500, 1e-6, 0.9, 500, 2]
    )
    cell100um = EPTfile(
        file2,
        "Protonated 20um",
        color='C5',
        thickness=0.620,
        circuit="R0-p(R1,CPE1)-p(R2,CPE2)-Ws1",
        initial_par=[1, 1000, 1e-9, 0.9, 20, 2e-6, 0.9, 400, 30]
    )
    cell1500um = EPTfile(
        file3,
        "Protonated 70um",
        color='C6',
        thickness=0.540,
        circuit="R0-p(R1,CPE1)-Ws1",
        initial_par=[1, 1500, 1e-8, 0.9, 500, 2]
    )

    def normalizer(values, cell: ept.Cell):
        return values * cell.area_mm2 * 1e-2

    files = [cell0um, cell100um, cell1500um]

    fig, ax = ept.create_fig()
    show_freq = True

    for i, file in enumerate(files):
        if file.ignore:
            continue
        cell_3mm = ept.Cell(3, file.thickness)
        print(file.path)
        file_path = path + file.path
        data = ept.load_data(file_path)
        print(f"Number of cycles: {len(data)}")

        for cycle in reversed(data):
            lines = cycle.plot_nyquist(
                ax,
                size=6,
                label=file.name,
                show_freq=show_freq,
                plot_range=(-20, 180),
                color=file.color,
                show_mark_label=False,
                cell=cell_3mm,
                normalize=normalizer,
                unit=r"$\Omega$.cm$^2$.mm"
            )
            # cycle.fit_nyquist(
            #     ax,
            #     file.circuit,
            #     file.initial_par,
            #     draw_circle=True,
            # )
            if show_freq:
                # marks = [lines[name][0] for name in lines.keys() if
                #          name.startswith('MP')]
                # print(marks)
                # leg = plt.legend(marks, ["1", "1", "1", "1", "1"], loc=4)
                # ax.add_artist(leg)
                # for artist in ax.artists:
                #     print(artist)
                show_freq = False
            break

    plt.show()


if __name__ == "__main__":
    print('Started')
    main()
    print('Finished')
