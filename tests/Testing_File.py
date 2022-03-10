import logging

import eclabfiles as ecf
import matplotlib.pyplot as plt


def setup_logger(logger):
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    f_string = "\x1b[37m"
    f_string += (
        "[%(asctime)s] %(name)s - %(module)s:%(lineno)d " "%(levelname)s:%(message)s"
    )
    f_string += "\x1b[0m"

    formatter = logging.Formatter(f_string, "%H:%m:%S")
    sh.setFormatter(formatter)

    def decorate_emit(fn):
        def new(*args):
            levelno = args[0].levelno
            if levelno >= logging.CRITICAL:
                color = "\x1b[31;1m"
            elif levelno >= logging.ERROR:
                color = "\x1b[31;2m"
            elif levelno >= logging.WARNING:
                color = "\x1b[33;1m"
            elif levelno >= logging.INFO:
                color = "\x1b[32;4m"
            elif levelno >= logging.DEBUG:
                color = "\x1b[34;1m"
            else:
                color = "\x1b[0m"
            # add colored *** in the beginning of the message
            args[0].levelname = f"{color}" + args[0].levelname + "\x1b[0m"
            args[0].msg = "\x1b[37m" + args[0].msg + "\x1b[0m"
            return fn(*args)

        return new

    sh.emit = decorate_emit(sh.emit)
    logger.addHandler(sh)


def main():
    path = r"C:\Users\ueli\Desktop\Sauter Ulrich\Projects\Thickness"
    file1 = (
        r"\20211217_B10P2_Water-1W_HT400C-3h_Li-3mm-280C"
        r"-30min_0um_EIS_01_PEIS_C15.mpr"
    )

    df = ecf.to_df(path + file1)
    df.plot(x="Re(Z)/Ohm", y="-Im(Z)/Ohm", kind="scatter")

    plt.show()


setup_logger(logging.getLogger("eisplottingtool"))
if __name__ == "__main__":
    print("start")
    main()
    print("stop")
