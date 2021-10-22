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
from typing import Union


class MarkPoint:
    """ Special point to mark during plotting.

    A mark point is given by a specific frequency. The mark point is described by a
    color and a name. A frequency range can be given to narrow the search area in
    frequency space for a data point.
    """

    def __init__(
            self,
            name: str,
            color: str,
            freq: float,
            delta_f: float = -1,
            ecr: bool = False
    ) -> None:
        """ Special point in the EIS spectrum

        @param name: Name of the mark point
        @param color: Color of the mark point
        @param freq: Specific frequency of the feature
        @param delta_f: interval to look for datapoints, if none given -> defualt is 10% of freq
        @param ecr: special value to mark ECR tail
        """
        self.name = name
        self.color = color
        self.freq = freq
        self.delta_f = delta_f
        if delta_f <= 0:
            self.delta_f = freq // 10
        self.ecr = ecr
        self.left = self.freq - self.delta_f  # left border of freq range
        self.right = self.freq + self.delta_f  # right border of freq range
        self.index = -1  # index of the first found data point matching in the freq range
        self.magnitude = np.floor(np.log10(freq))  # magnitude of the frequency


#  Some default markpoints
grain_boundaries = MarkPoint('LLZO-GB', 'blue', freq=3e5, delta_f=5e4)
hllzo = MarkPoint('HLLZO', 'orange', freq=3e4, delta_f=5e4)
lxzo = MarkPoint('LxZO', 'lime', freq=2e3, delta_f=5e2)
interface = MarkPoint('Interfacial resistance', 'magenta', freq=50, delta_f=5)
ecr_tail = MarkPoint('ECR', 'darkgreen', freq=0.5, delta_f=1, ecr=True)


class Cell:
    """
        Save the characteristics of a cell. Usefull for further calculation.
        TODO: Implement it :)
    """

    def __init__(self, diameter_mm, height_mm):
        """ Initializer of a cell

        @param diameter_mm: diameter of the cell in mm
        @param height_mm: height of the cell in mm
        """
        self.diameter_mm = diameter_mm
        self.height_mm = height_mm

        self.area_mm2 = (diameter_mm / 2) ** 2 * np.pi  # area of the cell
        self.volume_mm3 = self.area_mm2 * height_mm  # volume of the cell


class EISFrame:
    """ EISFrame used to store the data and plot/fit the data.

    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialises an EISFrame

        An EIS frame can plot a Nyquist plot and the lifecycle of the cell with different default settings.

        @param df: pandas.DataFrame containing the data
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
        """ Plots a Nyquist plot with the internal dataframe TODO: add all parameters to the function

        Plots a Nyquist plot with the internal dataframe. Will also mark the different markpoints on the plot.

        @param ax: matplotlib.axes.Axes to plot to
        @param image: path to image to include in plot
        @return: TODO: list of lines for markpoints/data, data line is first line
                 TODO: If image available also returns image axes and image
        """
        # check if the necessary data is available for a Nyquist plot
        if not {"freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm"}.issubset(self.df.columns):
            raise ValueError('Wrong data for a Nyquist Plot')

        # find indices of the markpoints. Takes first point that is in freq range
        for mark in self.mark_points:
            subsequent = (
                idx for idx, freq in enumerate(self.df["freq/Hz"])
                if mark.left < freq < mark.right)
            mark.index = next(subsequent, -1)

        # just so I dont have to write that all the time
        x_name = "Re(Z)/Ohm"
        y_name = "-Im(Z)/Ohm"

        # label for the plot
        x_label = r"Re(Z)/$\Omega$"
        y_label = r"-Im(Z)/$\Omega$"

        # check if any axes is given, if not GetCurrentAxis from matplotlib
        if ax is None:
            ax = plt.gca()

        # plot the data
        line = ax.plot(
            self.df[x_name],
            self.df[y_name],
            marker='o',
            color='black',
            ls='none',
        )
        lines = [line]  # store all the lines inside lines

        # plot each markpoint with corresponding color and name
        for mark in self.mark_points:
            line = ax.plot(
                self.df[x_name][mark.index],
                self.df[y_name][mark.index],
                marker='o',
                markerfacecolor=mark.color,
                markeredgecolor=mark.color,
                markersize=10,
                markeredgewidth=3,
                ls='none',
                label=mark.name)
            lines.append(line)

        # additional configuration for the plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(-50, None)
        ax.set_ylim(*ax.get_xlim())
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.locator_params(nbins=4)
        ax.set_aspect('equal')
        _plot_legend(ax)

        # if a path to a image is given, also plot it
        if image:
            imax = ax.inset_axes([.05, .5, .9, .2])
            img = plt.imread(image)
            imax.imshow(img)
            imax.axis('off')
            return lines, imax, img

        return lines

    def fit_nyquist(self, ax: axes.Axes, fit_circuit: str = '', fit_guess: str = '',
                    draw_circle: bool = True) -> list:
        """ Fitting for the nyquist TODO: add all options to the function

        @param ax: axes to draw the fit to
        @param fit_circuit: equivalence circuit for the fitting
        @param fit_guess: initial values for the fitting
        @param draw_circle: if the corresponding circles should be drawn or not
        @return: fitting parameters TODO: maybe more?
        """
        # load and prepare data
        frequencies = self.df["freq/Hz"]
        z = self.df["Re(Z)/Ohm"] - 1j * self.df["-Im(Z)/Ohm"]
        frequencies, z = preprocessing.ignoreBelowX(frequencies[3:], z[3:])
        frequencies = np.array(frequencies)
        z = np.array(z)
        # only for testing purposes like this
        circuit = fit_circuit
        if not fit_guess:
            fit_guess = [10, 1146.4, 3.5 * 1e-10, 1, 1210, .001, .5]
        if not fit_circuit:
            circuit = 'R_0-p(R_1,CPE_1)-p(R_2,CPE_2)'

        # bounds for the fitting
        bounds = ([0, 0, 1e-15, 0, 0, 1e-15, 0], [np.inf, np.inf, 1e12, 1, np.inf, 1e12, 1])

        # create the circuit and start the fitting still TODO: fix the global fitting routine
        custom_circuit = CustomCircuit(initial_guess=fit_guess, circuit=circuit, )
        custom_circuit.fit(frequencies, z, global_opt=True)

        # print the fitting parameters to the console
        print(custom_circuit)

        # plot the fitting result
        f_pred = np.logspace(-2, 7, 200)
        custom_circuit_fit = custom_circuit.predict(f_pred)
        line = ax.plot(
            np.real(custom_circuit_fit),
            -np.imag(custom_circuit_fit),
            label="fit",
            color="red",
            zorder=4)

        # check if circle needs to be drawn
        if draw_circle:
            self._plot_semis(custom_circuit, ax)

        _plot_legend(ax)
        return line

    def plot_lifecycle(self):
        # TODO plot lifecycle
        if {"time/s", "<Ewe>/V"}.issubset(self.df.columns):
            return True
        return False

    def _plot_semis(self, circuit: CustomCircuit, ax: axes.Axes = None):
        """ plots the semicircles to the corresponding circuit elements.

        the permitted elements are p(R,CPE), p(R,C) or TODO: any Warburg element
        @param circuit: CustomCircuit
        @param ax: axes to be plotted to
        """
        # check if axes is given, else get current axes
        if ax is None:
            ax = plt.gca()

        # read out details about circuit
        circuit_string = circuit.circuit
        names = circuit.get_param_names()[0]
        resistors = [names.index(name) for name in names if "R" in name]

        # split the circuit in to elements connected through series
        elements = circuit_string.split('-')
        for e in elements:
            # check if any of the elements is a parallel circuit
            if not e.startswith('p'):
                continue
            e = e.strip('p()')
            # TODO: Check if valid components
            # check if element is p(R,CPE) or P(R,C) if none skip to next elemenet
            if not (e.count('R') == 1 and (e.count('CPE') == 1 or e.count('C') == 1)):
                continue
            components = e.split(',')  # get the names of both elements

            # find the indices of the elements and all resistors that are in fornt of it
            components_index = [names.index(name) for name in names for component in components if component in name]
            prev_resistors = [resistor for resistor in resistors if resistor < min(components_index)]

            # get the fitted values
            components_values = circuit.parameters_[components_index]
            resistors_values = circuit.parameters_[prev_resistors]

            # calculate the data of the circle
            circle_data = predict_circle(*components_values) + np.sum(resistors_values)
            specific_freq = calc_specific_freq(*components_values)
            specific_freq_magnitude = np.floor(np.log10(specific_freq))
            color = 'black'
            ecr_color = next((m.color for m in self.mark_points if m.ecr), "green")
            # check with which markpoint the circle is associated by comparing magnitudes
            for mark in self.mark_points:
                if specific_freq_magnitude == mark.magnitude:
                    print(mark.name)
                    color = mark.color
                    break
                if specific_freq_magnitude <= 0:
                    print("ECR")
                    color = ecr_color
                    break
            # draw circle
            ax.fill_between(
                np.real(circle_data),
                -np.imag(circle_data),
                color=color,
                alpha=0.5,
                zorder=5,
                ls='None'
            )

        return


def create_fig(nrows: int = 1, ncols: int = 1, sharex='all', sharey='all', figsize=None, subplot_kw=None,
               gridspec_kw=None, **fig_kw) -> tuple[figure.Figure, Union[list[axes.Axes], axes.Axes]]:
    """ Creates the figure, axes for the plots and set the style of the plot
    
    @param sharex:
    @param sharey:
    @param figsize:
    @param subplot_kw:
    @param gridspec_kw:
    @param nrows: number of rows
    @param ncols: number of columns
    @return: the figure and list of created axes
    """
    set_plot_params()

    if not figsize:
        figsize = (6.4 * ncols, 4.8 * nrows)
    if not gridspec_kw:
        gridspec_kw = {"hspace": 0}
    elif gridspec_kw.get("hspace") is None:
        gridspec_kw["hspace"] = 0

    return plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw,
        subplot_kw=subplot_kw,
        **fig_kw
    )


def save_fig(path: str = '', fig: figure.Figure = None, show: bool = False, **kwargs) -> None:
    """ Saves the current figure at path


    @param path: path to save the figure
    @param fig: the figure to save
    @param show: show figure, no saving, False: save and show figure
    """
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    if not show:
        fig.savefig(path, bbox_inches='tight', **kwargs)
        fig.canvas.draw_idle()
    fig.show()


def calc_specific_freq(r, q, n=1):
    spec_frec = np.reciprocal(np.power(r * q, np.reciprocal(n)))
    spec_frec *= np.reciprocal(2 * np.pi)
    return spec_frec


def predict_circle(r, q, n, w=np.logspace(-2, 10, 200)) -> np.array:
    return np.reciprocal(1 / r + q * w ** n * np.exp(np.pi / 2 * n * 1j))


def set_plot_params() -> None:
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


def _plot_legend(ax: axes.Axes = None, loc='upper left', fontsize='small', frameon=False, markerscale=0.5,
                 handletextpad=0.1, mode='expand', **kwargs) -> legend.Legend:
    """ Adds legend to an axes

    @param ax: axes
    @param loc:
    @param fontsize:
    @param frameon:
    @param markerscale:
    @param handletextpad:
    @param mode:
    @param kwargs:
    @return: legend
    """
    if ax is None:
        ax = plt.gca()

    leg = ax.legend(
        loc=loc,
        fontsize=fontsize,
        frameon=frameon,
        markerscale=markerscale,
        handletextpad=handletextpad,
        mode=mode,
        **kwargs
    )
    return leg


def load_data(path: str, data_param: list) -> list['EISFrame']:
    """
        TODO: WIP
    """
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

    # TODO: Sort MB by technique
    # cycles = [mb[mb['cycle number'] == i] for i in range(int(max(mb['cycle number'])))]
    # sequences = []
    # for cycle in cycles:
    #     sequences.append([cycle[cycle['Ns'] == i] for i in range(int(max(cycle['Ns'])))])

    return cycles
