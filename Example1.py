import matplotlib.pyplot as plt
import numpy as np
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from scipy.optimize import Bounds

from Base import load_data, create_fig


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
    # https://stackoverflow.com/questions/43593592/errors-to-fit-parameters-of-scipy-optimize
    # https://stats.stackexchange.com/questions/71154/when-an-analytical-jacobian-is-available-is-it-better-to-approximate-the-hessia
    # https://stats.stackexchange.com/questions/350029/working-out-error-on-fit-parameters-for-nonlinear-fit

    z_fit = circuit.predict(frequencies)

    fig, axs = plt.subplots(1, 1)
    plot_nyquist(axs, z, fmt='o')
    plot_nyquist(axs, z_fit, fmt='-')

    plt.show()


def main2():
    data = load_data(r"Examples\ExampleData1.csv", sep=',')
    print(data.df.columns)
    fig, ax = create_fig()
    data.plot_nyquist(ax)
    data.fit_nyquist(ax, draw_circle=False, draw_circuit=False)

    plt.show()


class MyTakeStep:
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize
        self.rng = np.random.default_rng()

    def __call__(self, x):
        s = self.stepsize
        x[0:3] += self.rng.uniform(-s, s, x[0:3].shape)
        x[3] += self.rng.uniform(-0.01 * s, 0.01 * s)
        x[4:6] += self.rng.uniform(-s, s, x[4:6].shape)
        x[6] += self.rng.uniform(-0.01 * s, 0.01 * s)
        return x


if __name__ == "__main__":
    print("start")
    main2()
    print("end")
