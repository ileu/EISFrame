import matplotlib.pyplot as plt
import numpy as np
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from scipy.optimize import Bounds

from Base import load_data, create_fig
from Parser.CircuitParserCalc import calc_circuit


def main():
    frequencies, z = preprocessing.readCSV(r".\ExampleData1.csv")
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
    circuit = "R0-p(R1-Ws1,CPE1)"
    circuit2 = 'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1'
    freq = np.logspace(-4, 9, 500)
    symbols = ['o', 'd', 'x', 's']

    plt.plot()
    for i in range(4):
        param = {
            "R0": 100,
            "R1": 100,
            "R2": 100,
            "CPE1_0": 1e-9,
            "CPE1_1": 0.9,
            "Ws1_0": 200,
            "Ws1_1": 0.001 * 100 ** i
            }
        param2 = {'R0': 9.575122642822812, 'R1': 1853.4740741785354, 'CPE1_0': 1e-09, 'CPE1_1': 0.9985045295858299, 'R2': 2012.1396639033137, 'CPE2_0': 1e-07, 'CPE2_1': 1.0, 'Ws1_0': 1507.6154921209086, 'Ws1_1': 10.0}

        res = calc_circuit(param2, circuit2, freq)
        plt.scatter(
                res.real,
                -res.imag + 100 * i,
                c=freq * param2["Ws1_1"] / 2 / np.pi,
                marker=symbols[i],
                cmap='bwr',
                vmin=0,
                vmax=2,
                label=rf"$\tau$={10 ** i}s"
                )
        break
    plt.colorbar(label=r"$\omega\cdot\tau$ in multiples of $2\pi$")
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-10, 6000)
    ax.set_ylim(ax.get_xlim())
    plt.legend()
    plt.show()


def main3():
    data = load_data(
            r"C:\Users\ueli\Desktop\Sauter "
            r"Ulrich\Water-param\20210603_B6_water-4weeks-FC_01_PEIS_C03.mpr",
            sep=','
            )[-1]
    print("LOADED")
    circuit2 = 'R0-p(R1,CPE1)-p(R2,CPE2)-Ws1'
    param2 = [10.0, 1853.9, 3.6e-10, 1.0, 2012.5, 1.085e-8, 0.93477, 1507.7,
              1.9347]
    fig, ax = create_fig()
    data.plot_nyquist(ax)
    data.fit_nyquist(
            ax,
            fit_circuit=circuit2,
            fit_guess=param2,
            draw_circle=False,
            )
    print(param2)
    plt.show()


if __name__ == "__main__":
    print("start")
    main2()
    print("end")
