import timeit

import matplotlib.pyplot as plt
import numpy as np
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from scipy.optimize import Bounds

from eisplottingtool import load_data, create_fig, Cell
from eisplottingtool.parser import parse_circuit, circuit_components


def main1():
    frequencies, z = preprocessing.readCSV(r".\Examples\ExampleData1.csv")
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

    # low_res = results.lowest_optimization_result
    # ftol = 2.220446049250313e-09
    # tmp_i = np.zeros(len(low_res.x))
    # perror = []
    # for i in range(len(low_res.x)):
    #     tmp_i[i] = 1.0
    #     hess_inv_i = low_res.hess_inv(tmp_i)[i]
    #     uncertainty_i = np.sqrt(max(1, abs(low_res.fun)) * ftol * hess_inv_i)
    #     tmp_i[i] = 0.
    #     perror.append(uncertainty_i)
    # https://stackoverflow.com/questions/43593592/errors-to-fit-parameters
    # -of-scipy-optimize
    # https://stats.stackexchange.com/questions/71154/when-an-analytical
    # -jacobian-is-available-is-it-better-to-approximate-the-hessia
    # https://stats.stackexchange.com/questions/350029/working-out-error-on
    # -fit-parameters-for-nonlinear-fit

    z_fit = circuit.predict(frequencies)

    fig, axs = plt.subplots(1, 1)
    plot_nyquist(axs, z, fmt='o')
    plot_nyquist(axs, z_fit, fmt='-')

    plt.show()


def main2():
    tries = int(1e4)
    circuit = 'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1'
    names, calc = parse_circuit(circuit)
    p = {key: 1 for key in names.keys()}
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
    param2 = [0.0, 1037.9, 3.416e-10, 0.9896, 1512.9, 2.697e-8, 0.920, 743.7,
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
    param = dict(zip(info.keys(), param2))
    param["omega"] = np.logspace(-9, 9, 400)
    res = calc(param)
    res = res * Cell(3, 0.7).area_mm2 * 1e-2
    plt.plot(np.real(res), -np.imag(res), label="Initial Values")
    plt.show()


def main4():
    circuit2 = 'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1'
    param2 = [0.0, 1037.9, 3.416e-10, 1, 1512.9, 2.697e-8, 0.920, 743.7,
              2.78] # CPE1_n = 0.9896

    info, calc = parse_circuit(circuit2)
    names = [inf[0] for inf in info]
    pars = dict(zip(names, param2))
    print(1.0 / np.power(pars['R1'] * pars['CPE1_Q'], 1.0 / pars['CPE1_n']) / 2 / np.pi)
    print(1.0 / np.power(pars['R2'] * pars['CPE2_Q'], 1.0 / pars['CPE2_n']) / 2 / np.pi)
    data = load_data(
            r"G:\Collaborators\Sauter "
            r"Ulrich\Water-param\20210603_B6_water-4weeks-FC_01_PEIS_C03.mpr",
            sep=','
            )[-1]
    fig, axs = create_fig()
    data._plot_semis(circuit2, info, pars, ax=axs)
    plt.show()


if __name__ == "__main__":
    print("start")
    main4()
    print("end")
