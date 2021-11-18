import timeit

import matplotlib.pyplot as plt
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from scipy.optimize import Bounds

from Base import load_data, create_fig, Cell
from Parser.CircuitElements import circuit_components
from Parser.CircuitParser import parse_circuit



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
    circuit.predict()
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
    import numpy as np
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
    plt.show()


if __name__ == "__main__":
    print("start")
    for _ in range(50):
        main2()
    print("end")
