"""
TODO
"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler, axes, figure, legend
from matplotlib.ticker import AutoMinorLocator
from impedance.models.circuits import CustomCircuit
from impedance import preprocessing


class MarkPoint:
    """
    TODO
    """

    def __init__(
            self,
            name: str,
            color: str,
            freq: float,
            delta_f: float = -1,
            ecr: bool = False
    ) -> None:
        """Initialises a MarkPoint."""
        self.name = name
        self.color = color
        self.freq = freq
        self.delta_f = delta_f
        if delta_f <= 0:
            self.delta_f = freq // 10
        self.ecr = ecr
        self.left = self.freq - self.delta_f
        self.right = self.freq + self.delta_f
        self.index = -1
        self.magnitude = np.floor(np.log10(freq))


grain_boundaries = MarkPoint('LLZO-GB', 'blue', freq=3e5, delta_f=5e4)
hllzo = MarkPoint('HLLZO', 'orange', freq=3e4, delta_f=5e4)
lxzo = MarkPoint('LxZO', 'lime', freq=2e3, delta_f=5e2)
interface = MarkPoint('Interfacial resistance', 'magenta', freq=50, delta_f=5)
ecr_tail = MarkPoint('ECR', 'darkgreen', freq=0.5, delta_f=1, ecr=True)


class Cell:
    def __init__(self, diameter_mm, height_mm):
        self.diameter_mm = diameter_mm
        self.height_mm = height_mm

        self.area_mm2 = (diameter_mm / 2) ** 2 * np.pi
        self.volume_mm3 = self.area_mm2 * height_mm


class EISFrame:
    """
    TODO
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialises an EISFrame

        An EIS frame can plot a Nyquist plot and the lifecycle of the cell with different default settings.

        :param df: pandas.DataFrame containing the data
        """
        self.df = df
        self._default_mark_points = [grain_boundaries, hllzo, lxzo, interface, ecr_tail]
        self.mark_points = self._default_mark_points
        self.axes = {}  # TODO: put all plots in here

    def reset_markpoints(self) -> None:
        """ Resets list markpoints to default

        The default values of the list corresponds to the markpoints for
        grain boundaries, hllzo, lzo, interfacial resistance and ECR.
        """
        self.mark_points = self._default_mark_points

    def plot_nyquist(self, ax: axes.Axes = None, image: str = ''):
        """ Plots a Nyquist plot with the internal dataframe

        Plots a Nyquist plot with the internal dataframe. Will also mark the different markpoints on the plot.

        :param ax: matplotlib.axes.Axes to plot to
        :param image: path to image to include in plot

        :return: list of lines for markpoints/data, data line is first line
                 If image available also returns image axes and image
        """
        if not {"freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm"}.issubset(self.df.columns):
            raise ValueError('Wrong data for a Nyquist Plot')

        for mark in self.mark_points:
            subsequent = (
                idx for idx, freq in enumerate(self.df["freq/Hz"])
                if mark.left < freq < mark.right)
            mark.index = next(subsequent, -1)

        x_label = "Re(Z)/Ohm"
        y_label = "-Im(Z)/Ohm"

        if ax is None:
            ax = plt.gca()

        line = ax.plot(
            self.df[x_label],
            self.df[y_label],
            marker='o',
            color='black',
            ls='none',
        )
        lines = [line]

        for mark in self.mark_points:
            line = ax.plot(
                self.df[x_label][mark.index],
                self.df[y_label][mark.index],
                marker='o',
                markerfacecolor=mark.color,
                markeredgecolor=mark.color,
                markersize=10,
                markeredgewidth=3,
                ls='none',
                label=mark.name)
            lines.append(line)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(-50, None)
        ax.set_ylim(*ax.get_xlim())
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.locator_params(nbins=4)
        ax.set_aspect('equal')
        _plot_legend(ax)

        if image:
            imax = ax.inset_axes([.05, .5, .9, .2])
            img = plt.imread(image)
            imax.imshow(img)
            imax.axis('off')
            return lines, imax, img

        return lines

    def fit_nyquist(self, ax: axes.Axes, fit_guess: str = '', fit_circuit: str = '',
                    draw_circle: bool = True) -> None:
        """ Fitting for the nyquist

        :param ax:
        :param fit_guess: initial values for the fitting
        :param fit_circuit: equivalence circuit for the fitting
        :param draw_circle:
        :return:
        """
        for mark in self.mark_points:
            print(mark.name, mark.magnitude)
        frequencies = self.df["freq/Hz"]
        z = self.df["Re(Z)/Ohm"] - 1j * self.df["-Im(Z)/Ohm"]
        frequencies, z = preprocessing.ignoreBelowX(frequencies[3:], z[3:])
        frequencies = np.array(frequencies)
        z = np.array(z)
        circuit = fit_circuit
        if not fit_guess:
            fit_guess = [10, 1146.4, 3.5 * 1e-10, 1, 1210, .001, .5]
        if not fit_circuit:
            circuit = 'R_0-p(R_1,CPE_1)-p(R_2,CPE_2)'

        bounds = ([0, 0, 1e-15, 0, 0, 1e-15, 0], [np.inf, np.inf, 1e12, 1, np.inf, 1e12, 1])
        bounds = ([1, 1, 1, 0, 1, 1, 0], [3, 3, 4, 1, 5, 5, 1])

        custom_circuit = CustomCircuit(initial_guess=fit_guess, circuit=circuit, )
        custom_circuit.fit(frequencies, z, global_opt=True)
        print(custom_circuit)

        f_pred = np.logspace(-2, 7, 200)
        custom_circuit_fit = custom_circuit.predict(f_pred)
        line = ax.plot(
            np.real(custom_circuit_fit),
            -np.imag(custom_circuit_fit),
            label="fit",
            color="red",
            zorder=4)

        if draw_circle:
            self._plot_semis(custom_circuit, ax)

        _plot_legend(ax)
        return line

    def plot_lifecycle(self):
        # TODO
        if {"time/s", "<Ewe>/V"}.issubset(self.df.columns):
            return True
        return

    def _plot_semis(self, circuit: CustomCircuit, ax: axes.Axes = None):
        """TODO

        :param self:
        :param circuit:
        :param ax:
        :return:
        """
        if ax is None:
            ax = plt.gca()

        circuit_string = circuit.circuit
        names = circuit.get_param_names()[0]
        resistors = [names.index(name) for name in names if "R" in name]

        elements = circuit_string.split('-')
        for e in elements:
            if not e.startswith('p'):
                continue
            e = e.strip('p()')
            # TODO: Check if valid components
            if not (e.count('R') == 1 and (e.count('CPE') == 1 or e.count('C') == 1)):
                continue
            components = e.split(',')

            components_index = [names.index(name) for name in names for component in components if component in name]
            prev_resistors = [resistor for resistor in resistors if resistor < min(components_index)]

            components_values = circuit.parameters_[components_index]
            resistors_values = circuit.parameters_[prev_resistors]

            circle_data = predict_circle(*components_values) + np.sum(resistors_values)
            specific_freq = calc_specific_freq(*components_values)
            specific_freq_magnitude = np.floor(np.log10(specific_freq))
            color = 'black'
            ecr_color = next((m.color for m in self.mark_points if m.ecr), "green")
            for mark in self.mark_points:
                if specific_freq_magnitude == mark.magnitude:
                    print(mark.name)
                    color = mark.color
                    break
                if specific_freq_magnitude <= 0:
                    print("ECR")
                    color = ecr_color
                    break
            ax.fill_between(
                np.real(circle_data),
                -np.imag(circle_data),
                color=color,
                alpha=0.5,
                zorder=5,
                ls='None'
            )

        return


def _circle_interpretation(self, circuit, ax):
    return


def create_fig(nrows: int = 1, ncols: int = 1) -> tuple[figure.Figure, list]:
    """ Creates the figure, axes for the plots and set the style of the plot

    :param nrows: number of rows
    :param ncols: number of columns
    :return: the figure and list of created axes
    """
    set_plot_params()
    return plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 4.8 * nrows),
        sharex='all',
        sharey='all',
        gridspec_kw={"hspace": 0},
    )


def save_fig(path: str = '', fig: figure.Figure = None, show: bool = False):
    """ Saves the current figure at path

    :param path: path to save the figure
    :param fig: the figure to save
    :param show: True: show figure, no saving, False: save and show figure
    """
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    if not show:
        fig.savefig(path, bbox_inches='tight')
        fig.canvas.draw_idle()
    fig.show()


def calc_specific_freq(r, q, n=1):
    spec_frec = np.reciprocal(np.power(r * q, np.reciprocal(n)))
    spec_frec *= np.reciprocal(2 * np.pi)
    return spec_frec


def predict_circle(r, q, n, w=np.logspace(-2, 10, 200)) -> np.array:
    return np.reciprocal(1 / r + q * w ** n * np.exp(np.pi / 2 * n * 1j))


def set_plot_params() -> None:
    """

    :return:
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 10
    rcParams['axes.linewidth'] = 1.1
    rcParams['axes.labelpad'] = 10.0
    plot_color_cycle = cycler(
        'color',
        [
            '000000',
            '0000FE',
            'FE0000',
            '008001',
            'FD8000',
            '8c564b',
            'e377c2',
            '7f7f7f',
            'bcbd22',
            '17becf',
        ])
    rcParams['axes.prop_cycle'] = plot_color_cycle
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0
    rcParams.update({
        "figure.subplot.hspace": 0,
        "figure.subplot.left": 0.11,
        "figure.subplot.right": 0.946,
        "figure.subplot.bottom": 0.156,
        "figure.subplot.top": 0.965,
        "xtick.major.size": 4,
        "xtick.minor.size": 2.5,
        "xtick.major.width": 1.1,
        "xtick.minor.width": 1.1,
        "xtick.major.pad": 5,
        "xtick.minor.visible": True,
        "xtick.direction": 'in',
        "xtick.top": True,
        "ytick.major.size": 4,
        "ytick.minor.size": 2.5,
        "ytick.major.width": 1.1,
        "ytick.minor.width": 1.1,
        "ytick.major.pad": 5,
        "ytick.minor.visible": True,
        "ytick.direction": 'in',
        "ytick.right": True,
        "lines.markersize": 10,
        "lines.markeredgewidth": 0.8,
    })


def _plot_legend(ax: axes.Axes = None) -> legend.Legend:
    """ Adds legend to an axes

    :param ax: axes
    :return: legend
    """
    if ax is None:
        ax = plt.gca()

    leg = ax.legend(
        loc='upper left',
        fontsize='small',
        frameon=False,
        markerscale=0.5,
        handletextpad=0.1,
        mode='expand')
    return leg


def load_data(path: str, data_param: list) -> list['EISFrame']:
    if "mpt" not in path:
        raise ValueError("Only mpt file supported atm")
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
        cycles[i] = EISFrame(cycles[i][data_param])

    return cycles
