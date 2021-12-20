import re
import timeit

import matplotlib.pyplot as plt
import numpy as np
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from scipy.optimize import Bounds, fminbound

from eisplottingtool import load_data, create_fig, Cell, save_fig
from eisplottingtool.parser import parse_circuit, circuit_components

# https://stackoverflow.com/questions/43593592/errors-to-fit-parameters
# -of-scipy-optimize
# https://stats.stackexchange.com/questions/71154/when-an-analytical
# -jacobian-is-available-is-it-better-to-approximate-the-hessia
# https://stats.stackexchange.com/questions/350029/working-out-error-on
# -fit-parameters-for-nonlinear-fit

def main1():
    frequencies, z = preprocessing.readCSV(r"../Examples/ExampleData1.csv")
    frequencies, z = preprocessing.ignoreBelowX(frequencies, z)

    circuit = CustomCircuit(
            'R0-p(R1,C1)-p(R2-Wo1,C2)',
            initial_guess=[.01, .01, 100, .01, .05, 100, 1]
            )
    bounds = ([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
              [10, 1, 10, 1, 10, 10000, 1000])

    circuit.fit(
            frequencies, z, global_opt=True, bounds=bounds,
            minimizer_kwargs={"bounds": Bounds(*bounds), "method": 'L-BFGS-B'},
            niter=10
            )
    print(circuit)
    z_fit = circuit.predict(frequencies)

    fig, axs = plt.subplots(1, 1)
    plot_nyquist(axs, z, fmt='o')
    plot_nyquist(axs, z_fit, fmt='-')

    plt.show()


def main2():
    tries = int(1e4)
    circuit = 'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1'
    infos, calc = parse_circuit(circuit)
    p = {info[0]: 1 for info in infos}
    custom_circuit = CustomCircuit(
            circuit,
            initial_guess=p.values()
            )
    p["omega"] = 1
    p2 = circuit_components
    p2["param"] = p

    res1 = timeit.timeit('calc(p)', number=tries, globals=locals())
    res4 = timeit.timeit(
            'custom_circuit.predict(np.array([1]),True)',
            number=tries,
            globals=locals()
            )

    print(str(res1) + "\t" + str(res4))


def main3():
    data = load_data(
            r"G:\Collaborators\Sauter "
            r"Ulrich\Water-param\20210603_B6_water-4weeks-FC_01_PEIS_C03.mpr",
            sep=','
            )[-1]
    print("LOADED")
    circuit2 = 'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1'
    param2 = [0.1, 1037.9, 3.416e-10, 0.9896, 1512.9, 2.697e-8, 0.920, 743.7,
              2.78]
    fig, ax = create_fig()
    data.plot_nyquist(ax, cell=Cell(3, 0.7))
    print("PLOTTED")
    data.fit_nyquist(
            ax,
            fit_circuit=circuit2,
            fit_guess=param2,
            draw_circle=False,
            cell=Cell(3, 0.7)
            )
    info, calc = parse_circuit(circuit2)
    names = [inf[0] for inf in info]
    param = dict(zip(names, param2))
    res = calc(param, np.logspace(-9, 9, 400))
    res = res * Cell(3, 0.7).area_mm2 * 1e-2
    plt.plot(np.real(res), -np.imag(res), label="Initial Values")
    plt.show()


def main4():
    path = r"C:\Users\ueli\Desktop\Sauter Ulrich\EIS and cycling raw data"
    file1 = r"\20200422_LLZTO_polished_water21h_400C-3h_IR_01_PEIS_C04.mpr"
    file2 = r"\20200424_LLZTO_polished_Ethnaol-15min-600C_PEIS_C13.mpr"
    file3 = r"\20200528_LLZTO_polished_Batch3-Ampcera-acetone-2h_400C" \
            r"-Ar_01_PEIS_C14.mpr"
    circuit = 'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1'
    param = [0.1, 1037.9, 3.416e-10, 0.9, 1512.9, 2.697e-8, 0.9, 743.7, 2.78]
    files = [file2]
    for file in files:
        print(file)
        name = re.search(r'_.*?C', file).group(0)[1:]
        data = load_data(
                path + file,
                sep=','
                )[-1]
        fig, axs = create_fig()
        data.plot_nyquist(axs)
        data.fit_nyquist(
                axs,
                circuit,
                param,
                path=path + rf"\plots\{name}_fit.txt"
                )

        save_fig(path + rf"\plots\{name}_EIS-Plot.png")


def main5():
    circuit = 'p(R1,CPE1)'
    param = {'R1': 1037.9, 'CPE1_Q': 3.416e-10, 'CPE1_n': 0.9}
    w_max = (param['R1'] * param['CPE1_Q']) ** (- 1.0 / param['CPE1_n'])
    f_max = w_max / 2 / np.pi

    elem_info, elem_eval = parse_circuit(circuit)
    max_x = fminbound(
            lambda x: np.imag(elem_eval(param, x)),
            1,
            1e12, xtol=1e-9, maxfun=1000,
            )
    print(5 * '*', " Calc ", 5 * '*')
    print(f_max)
    print(elem_eval(param, f_max))
    print(5 * '*', " Maximize  ", 5 * '*')
    print(max_x)
    print(elem_eval(param, max_x))


if __name__ == "__main__":
    print("start")
    main4()
    print("end")
