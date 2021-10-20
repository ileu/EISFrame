import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from scipy.optimize import Bounds
import eclabfiles as ecf

import Base


def main():
    data_path = r"G:\Limit\VMP3 data\Ueli\Ampcera-Batch8\20210916_B8-P1_HT750C-3h_Li-3mm-300C-30min\20210916_B8-P1_HT750C-3h_Li-3mm-300C-30min_EIS_01_PEIS_C05.mpr"
    image_path = r"G:\Collaborators\Sauter Ulrich\Ampcera-Batch8P1\References\Images\Circuit_750C.png"
    data_param = ["time/s", "<Ewe>/V", "freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm"]

    test123 = ecf.to_df(data_path)
    print(test123)

    frequencies, Z = preprocessing.readCSV(r".\ExampleData1.csv")
    frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)

    circuit = CustomCircuit('R0-p(R1,C1)-p(R2-Wo1,C2)', initial_guess=[.01, .01, 100, .01, .05, 100, 1])
    bounds = ([0, 0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    circuit.fit(frequencies, Z, global_opt=True, bounds=bounds, minimizer_kwargs={"bounds": Bounds(*bounds)}, niter=300,
                callback=my_callback)
    Z_fit = circuit.predict(frequencies)

    fig, axs = Base.create_fig(1, 1)
    plot_nyquist(axs, Z, fmt='o')
    plot_nyquist(axs, Z_fit, fmt='-')

    plt.show()


def my_callback(x, f, accept):
    if accept:
        print(x, f)


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


def my_callback(x, f, accept):
    print(x, f, accept)


def import_data_to_frame(path: str, data_param: list):
    mpt_file = open(path, 'r')
    next(mpt_file)
    num_headers_re = re.compile(r'Nb header lines : (?P<num>\d+)\s*$')
    num_headers_match = num_headers_re.match(next(mpt_file))
    num_headers = int(num_headers_match['num'])
    for __ in range(num_headers - 3):
        next(mpt_file)
    data = pd.read_csv(mpt_file, sep='\t', encoding='windows-1252')

    cycles = []
    for i in range(int(max(data['cycle number']))):
        cycles.append(data[data['cycle number'] == i])
        cycles[i] = cycles[i][data_param]

    return cycles


def import_csv_to_frame(path: str, data_param: list, sep='\t'):
    data = pd.read_csv(path, sep=sep, encoding='unicode_escape')
    return Base.EISFrame(data[data_param])


def calc(params, w):
    z1 = params[0]
    y1 = 1 / params[1] + 1j * w * params[2]
    z2_t = 1j * w * params[5]
    z2 = params[3] + params[4] / np.tanh(z2_t) / np.sqrt(z2_t)
    y2 = 1j * w * params[6] + 1 / z2
    return z1 + 1 / y1 + 1 / y2


if __name__ == "__main__":
    print("start")
    main()
    print("end")

'''
# Input three data files into three worksheets within one workbook
wb = op.new_book()
wb.set_int('nLayers',3) # Set number of sheets
for wks, fn in zip(wb, ['S15-125-03.dat', 'S21-235-07.dat', 'S32-014-04.dat']):
    wks.from_file(os.path.join(op.path('e'), 'Samples', 'Import and Export', fn))

# Add data plots onto the graph
gp = op.new_graph(template='PAN2VERT')  # load Vertical 2 Panel graph template

# Loop over layers and worksheets to add individual curve.
for i, gl in enumerate(gp):
    for wks in wb:
        plot = gl.add_plot(wks,1+i)
    gl.group()
    gl.rescale()

# Customize legend
lgnd = gp[1].label('Legend')
lgnd.text='\l(1) %(1, @ws)\n\l(2) %(2, @ws)\n\l(3) %(3, @ws)'
lgnd.set_int('left',4900)
lgnd.set_int('top',100)

gp[0].label('Legend').remove()
'''