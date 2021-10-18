import time

from impedance.models.circuits import CustomCircuit

from Help2 import wrapCircuit, wrapCircuit2
from Example1 import import_txt_to_frame
from impedance import preprocessing
import numpy as np
from scipy.optimize import basinhopping, Bounds

data_path = r"C:\Users\ueli\Desktop\Sauter Ulrich\20210920_B8-P1_HT400C-3h_Li-3mm-300C-30min_FCandPT_01_PEIS_C02.txt"
image_path = r"C:\Users\ueli\Desktop\Sauter Ulrich\Ampcera-Batch8P1\References\Images\Circuit_400C.png"

data_param = ["time/s", "<Ewe>/V", "freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm"]

test = import_txt_to_frame(data_path, data_param)

circuit = 'R_0-p(R_1,CPE_1)-p(R_2,CPE_2)'
constants = {}
Z = test.df["Re(Z)/Ohm"] - 1j * test.df["-Im(Z)/Ohm"]
frequencies = test.df["freq/Hz"]

frequencies, Z = preprocessing.readCSV(r"C:\Users\ueli\Desktop\exampleData.csv")

frequencies, Z = preprocessing.ignoreBelowX(frequencies[3:], Z[3:])
Z = np.array(Z)

bounds_minimizer = [(1e-32, None), (1e-32, None), (1e-32, None), (0, 1), (1e-32, None), (1e-32, None), (0, 1)]
bounds = Bounds([1e-32, 1e-32, 1e-32, 1e-32, 1e-32, 1e-32, 1e-32], [np.inf, np.inf, np.inf, 1, np.inf, np.inf, 1])
initial_guess = [10, 1146.4, 3.5 * 1e-10, 1, 1210, .001, .5]
parameters1 = [1.45016889e+01, 1.21769690e+03, 1.08569271e-03, 5.36309512e-01, 1.32171999e+03, 7.16204638e-10,
               9.44842206e-01]

circuit = 'R0-p(R1,C1)-p(R2-Wo1,C2)'
initial_guess = [.01, .01, 100, .01, .05, 100, 1]


def rmse(a, b):
    """
    A function which calculates the root mean squared error
    between two vectors.

    Notes
    ---------
    .. math::

        RMSE = \\sqrt{\\frac{1}{n}(a-b)^2}
    """

    n = len(a)
    return np.linalg.norm(a - b) / np.sqrt(n)


def opt_function(x):
    """ Short function for basinhopping to optimize over.
    We want to minimize the RMSE between the fit and the data.

    Parameters
    ----------
    x : args
        Parameters for optimization.

    Returns
    -------
    function
        Returns a function (RMSE as a function of parameters).
    """
    return rmse(wrapCircuit(circuit, constants)(frequencies, *x),
                np.hstack([Z.real, Z.imag]))


class BasinhoppingBounds(object):
    """ Adapted from the basinhopping documetation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
    """

    def __init__(self, xmin, xmax):
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)

    def __call__(self, **kwargs):
        x = kwargs['x_new']
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


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
    # print(wrapCircuit(circuit, constants))
    # x = wrapCircuit2(circuit, constants)(frequencies, *parameters1)
    # print(x)
    # plt.figure()
    # plt.plot(x.real, -x.imag)
    # plt.plot(Z.real, -Z.imag)
    # plt.show()
    basinhopping_bounds = BasinhoppingBounds(xmin=bounds.lb,
                                             xmax=bounds.ub)
    mytakestep = MyTakeStep()

    start = time.time()
    results = basinhopping(opt_function, x0=initial_guess,
                           accept_test=basinhopping_bounds,
                           minimizer_kwargs={"bounds": bounds_minimizer, "method": 'L-BFGS-B'},
                           niter=100, disp=False)
    end = time.time()
    popt = results.x
    print(results)
    print(end - start)
    print(popt)
    cir = CustomCircuit(circuit)

