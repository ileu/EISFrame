import logging

logging.basicConfig(level=logging.INFO)


class EPTfile:
    def __init__(self, path, name, ignore=False):
        self.ignore = ignore
        self.name = name
        self.path = path
        self.color = None


def main():
    path = r"G:\Collaborators\Sauter Ulrich\Projects\Thickness"
    file1 = r"\20211217_B10P2_Water-1W_HT400C-3h_Li-3mm-280C-30min_0um_EIS_01_PEIS_C15.mpr"
    file2 = r"\20211216_B10P2_Water-1W_HT400C-3h_Li-3mm-280C-30min_100um_EIS_01_PEIS_C04.mpr"
    file3 = r"\20211216_B10P2_Water-1W_HT400C-3h_Li-3mm-280C-30min_150um_EIS_01_PEIS_C13.mpr"



    pass


if __name__ == "__main__":
    logging.info('Started')
    main()
    logging.info('Finished')
