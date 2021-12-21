import logging
import eisplottingtool as ept
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)


class EPTfile:
    def __init__(self, path, name, ignore=False, thickness=1.0):
        self.ignore = ignore
        self.name = name
        self.path = path
        self.thickness = thickness
        self.color = None


def main():
    path = r"G:\Collaborators\Sauter Ulrich\Projects\Thickness"
    file1 = r"\20211217_B10P2_Water-1W_HT400C-3h_Li-3mm-280C-30min_0um_EIS_01_PEIS_C15.mpr"
    file2 = r"\20211216_B10P2_Water-1W_HT400C-3h_Li-3mm-280C-30min_100um_EIS_01_PEIS_C04.mpr"
    file3 = r"\20211216_B10P2_Water-1W_HT400C-3h_Li-3mm-280C-30min_150um_EIS_01_PEIS_C13.mpr"

    cell0um = EPTfile(file1, "Protonated ref")
    cell100um = EPTfile(file2, "Protonated 100um", thickness=0.699)
    cell1500um = EPTfile(file3, "Protonated 150um", thickness=0.699)

    files = [cell0um, cell100um, cell1500um]

    cell_3mm = ept.Cell(3, 0.7)
    fig, ax = ept.create_fig()
    show_freq = True

    for i, file in enumerate(files):
        if file.ignore:
            continue
        print(file.path)
        file_path = path + file.path
        data = ept.load_data(file_path)
        print(f"Number of cycles: {len(data)}")

        for cycle in data:
            cycle.mark_points = []
            cycle.plot_nyquist(
                    ax,
                    size=6,
                    cell=cell_3mm,
                    label=file.name,
                    show_freq=show_freq,
                    plot_range=(-20, 250),
                    color=file.color
                    )
            show_freq = False
            break
    plt.show()


if __name__ == "__main__":
    logging.info('Started')
    main()
    logging.info('Finished')
