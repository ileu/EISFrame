"""
    This Module has been made for automated plotting of EIS data and fitting.
    The main object is called EISFrame which includes most of the features
    Author: Ueli Sauter
    Date last edited: 25.10.2021
    Python Version: 3.9.7
"""
import os
import warnings
from typing import Union

import eclabfiles as ecf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from matplotlib import rcParams, cycler, axes, figure, legend
from matplotlib.ticker import AutoMinorLocator


class MarkPoint:
    """ Special point to mark during plotting.

    A mark point is given by a specific frequency. The mark point is
    described by a
    color and a name. A frequency range can be given to narrow the search
    area in
    frequency space for a data point.
    """

    def __init__(
            self, name: str, color: str, freq: float, delta_f: float = -1,
            ecr: bool = False
    ) -> None:
        """ Special point in the EIS spectrum

        @param name: Name of the mark point
        @param color: Color of the mark point
        @param freq: Specific frequency of the feature
        @param delta_f: interval to look for datapoints, if none given ->
        defualt is 10% of freq
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
        self.index = -1  # index of the first found data point matching in
        # the freq range
        self.magnitude = np.floor(np.log10(freq))  # magnitude of the frequency

    def __repr__(self):
        out = f"{self.name} @ {self.freq}, color {self.color}"
        out += f"with index {self.index}"
        return out


#  Some default mark points
grain_boundaries = MarkPoint('LLZO-GB', 'blue', freq=3e5, delta_f=5e4)
hllzo = MarkPoint('HLLZO', 'orange', freq=3e4, delta_f=5e3)
lxlzo = MarkPoint('LxLZO', 'lime', freq=2e3, delta_f=5e2)
interface = MarkPoint('Interfacial resistance', 'magenta', freq=50, delta_f=5)
ecr_tail = MarkPoint('ECR', 'darkgreen', freq=0.5, delta_f=1, ecr=True)


class Cell:
    """
        Save the characteristics of a cell. Usefull for further calculation.
    """

    def __init__(self, diameter_mm, thickness_mm):
        """ Initializer of a cell

        @param diameter_mm: diameter of the cell in mm
        @param thickness_mm: height of the cell in mm
        """
        self.diameter_mm = diameter_mm
        self.height_mm = thickness_mm

        self.area_mm2 = (diameter_mm / 2) ** 2 * np.pi  # area of the cell
        self.volume_mm3 = self.area_mm2 * thickness_mm  # volume of the cell

    def __repr__(self):
        return f"dia: {self.diameter_mm}, area: {self.area_mm2}"


class EISFrame:
    """ EISFrame used to store the data and plot/fit the data.

    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialises an EISFrame

        An EIS frame can plot a Nyquist plot and the lifecycle of the cell
        with different default settings.

        @param df: pandas.DataFrame containing the data
        """
        self.df = df
        self._default_mark_points = [grain_boundaries, hllzo, lxlzo, interface,
                                     ecr_tail]
        self.mark_points = self._default_mark_points
        self.lines = {}

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()

    def reset_markpoints(self) -> None:
        """ Resets list markpoints to default

        The default values of the list corresponds to the markpoints for
        grain boundaries, hllzo, lzo, interfacial resistance and ECR.
        """
        self.mark_points = self._default_mark_points

    def plot_nyquist(
            self, ax: axes.Axes = None, image: str = '', cell: Cell = None,
            exclude_start: int = None, exclude_end: int = None, ls='None',
            marker='o', plot_range=None, label=None, size=8, scale=1.5
    ):
        """ Plots a Nyquist plot with the internal dataframe

        Plots a Nyquist plot with the internal dataframe. Will also mark the
        different markpoints on the plot.

        @param ax: matplotlib.axes.Axes to plot to
        @param image: path to image to include in plot
        @param cell:
        @param exclude_start:
        @param exclude_end:
        @param ls:
        @param marker:
        @param plot_range:
        @param label:
        @param size:
        @param scale:
        @return: dictionary of matplotlib.lines.Line2D containing all the
        drawn plots
        """
        # check if the necessary data is available for a Nyquist plot
        if not {"freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm"}.issubset(
                self.df.columns
        ):
            warnings.warn('Wrong data for a Nyquist Plot', RuntimeWarning)
            return

        # label for the plot
        x_label = r"Re(Z)/$\Omega$"
        y_label = r"-Im(Z)/$\Omega$"

        # check if any axes is given, if not GetCurrentAxis from matplotlib
        if ax is None:
            ax = plt.gca()

        # remove all data points with (0,0)
        df = self.df.drop(self.df[self.df["Re(Z)/Ohm"] == 0].index)

        x_data = df["Re(Z)/Ohm"][slice(exclude_start, exclude_end)]
        y_data = df["-Im(Z)/Ohm"][slice(exclude_start, exclude_end)]

        if cell is not None:
            x_data = x_data * cell.area_mm2 * 1e-2
            x_label += r"$.cm^2$"

            y_data = y_data * cell.area_mm2 * 1e-2
            y_label += r"$.cm^2$"

        # find indices of the mark points. Takes first point that is in freq
        # range
        for mark in self.mark_points:
            subsequent = (idx for idx, freq in enumerate(self.df["freq/Hz"]) if
                          mark.left < freq < mark.right)
            mark.index = next(subsequent, -1)

        # plot the data
        line = ax.plot(
                x_data, y_data, marker=marker, ls=ls, label=label,
                markersize=size, )
        lines = {"Data": line}  # store all the lines inside lines

        # plot each mark point with corresponding color and name
        for mark in self.mark_points:
            if mark.index < 0:
                continue
            line = ax.plot(
                    x_data[mark.index], y_data[mark.index], marker='o',
                    markerfacecolor=mark.color, markeredgecolor=mark.color,
                    markersize=scale * size, markeredgewidth=3, ls='none',
                    label=mark.name
            )
            lines[mark.name] = line

        # additional configuration for the plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plot_range is None:
            ax.set_xlim(-max(x_data) * 0.05, max(x_data) * 1.05)
        else:
            ax.set_xlim(*plot_range)

        ax.set_ylim(*ax.get_xlim())
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.locator_params(nbins=4)
        ax.set_aspect('equal')
        _plot_legend(ax)

        # add lines to the axes property
        self.lines.update(lines)

        # if a path to a image is given, also plot it
        if image:
            imax = ax.inset_axes([.05, .5, .9, .2])
            img = plt.imread(image)
            imax.imshow(img)
            imax.axis('off')
            lines["Image"] = img
            return lines, imax

        return lines

    def fit_nyquist(
            self, ax: axes.Axes, fit_circuit: str = '', fit_guess: str = '',
            fit_bounds: tuple = None, global_opt: bool = False,
            draw_circle: bool = True, draw_circuit: bool = False
    ) -> tuple[list, dict]:
        """ Fitting for the nyquist TODO: add all options to the function

        @param ax: axes to draw the fit to
        @param fit_circuit: equivalence circuit for the fitting
        @param fit_guess: initial values for the fitting
        @param fit_bounds:
        @param global_opt:
        @param draw_circle: if the corresponding circles should be drawn or not
        @param draw_circuit:
        @return: fitting parameters
        """
        # load and prepare data
        frequencies = self.df["freq/Hz"]
        z = self.df["Re(Z)/Ohm"] - 1j * self.df["-Im(Z)/Ohm"]
        frequencies, z = preprocessing.ignoreBelowX(frequencies[3:], z[3:])
        frequencies = np.array(frequencies)
        z = np.array(z)
        # only for testing purposes like this
        if fit_guess is None:
            fit_guess = [10, 1146.4, 3.5 * 1e-10, 1, 1210, .001, .5]
        if fit_circuit is None:
            fit_circuit = 'R_0-p(R_1,CPE_1)-p(R_2,CPE_2)'

        # bounds for the fitting
        if fit_bounds is None:
            bounds = ([0, 0, 1e-15, 0, 0, 1e-15, 0],
                      [np.inf, np.inf, 1e12, 1, np.inf, 1e12, 1])

        # create the circuit and start the fitting still
        circuit = CustomCircuit(initial_guess=fit_guess, circuit=fit_circuit)
        circuit.fit(frequencies, z, global_opt=global_opt)

        # print the fitting parameters to the console
        print(circuit)

        parameters = dict(zip(circuit.get_param_names(), circuit.parameters_))

        # plot the fitting result
        f_pred = np.logspace(-2, 7, 200)
        custom_circuit_fit = circuit.predict(f_pred)
        line = ax.plot(
                np.real(custom_circuit_fit), -np.imag(custom_circuit_fit),
                label="fit", color="red", zorder=4
        )

        self.lines["fit"] = line
        # check if circle needs to be drawn
        if draw_circle:
            self._plot_semis(circuit, ax)

        if draw_circuit:
            self._plot_circuit(circuit, ax)

        _plot_legend(ax)
        return line, parameters

    def plot_lifecycle(self):
        # TODO plot lifecycle
        if {"time/s", "<Ewe>/V"}.issubset(self.df.columns):
            warnings.warn('Wrong data for a Nyquist Plot', RuntimeWarning)
            return
        return

    def _plot_semis(self, circuit: CustomCircuit, ax: axes.Axes = None):
        """ plots the semicircles to the corresponding circuit elements.

        the permitted elements are p(R,CPE), p(R,C) or
        TODO: any Warburg element
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
            # check if element is p(R,CPE) or P(R,C) if none skip to next
            # element
            if not (e.count('R') == 1 and (
                    e.count('CPE') == 1 or e.count('C') == 1)):
                continue
            components = e.split(',')  # get the names of both elements

            # find the indices of the elements and all resistors that are in
            # front of it
            components_index = [names.index(name) for name in names for
                                component in components if component in name]
            prev_resistors = [resistor for resistor in resistors if
                              resistor < min(components_index)]

            # get the fitted values
            components_values = circuit.parameters_[components_index]
            resistors_values = circuit.parameters_[prev_resistors]

            # calculate the data of the circle
            circle_data = predict_circle(*components_values) + np.sum(
                    resistors_values
            )
            specific_freq = calc_specific_freq(*components_values)
            specific_freq_magnitude = np.floor(np.log10(specific_freq))
            color = 'black'
            ecr_color = next(
                    (m.color for m in self.mark_points if m.ecr), "green"
            )
            # check with which mark point the circle is associated by
            # comparing magnitudes
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
                    np.real(circle_data), -np.imag(circle_data), color=color,
                    alpha=0.5, zorder=5, ls='None'
            )

        return

    def _plot_circuit(self, circuit: CustomCircuit, ax: axes.Axes = None):
        # TODO: Look at SchemDraw to draw circuit and color with different
        #  mark  points

        # check if axes is given, else get current axes
        if ax is None:
            ax = plt.gca()

        # read out details about circuit
        circuit_string = circuit.circuit
        names = circuit.get_param_names()[0]

        elements = circuit_string.split('-')
        # count how many components in string -> n
        # if n%2  == 1 -> draw one element in center and do for (n-1) i.e.
        # even n
        # if n%2 == 0 -> draw elements moved by 0.5 (maybe 0.25?) up and down
        return


def create_fig(
        nrows: int = 1, ncols: int = 1, sharex='all', sharey='all',
        figsize=None, subplot_kw=None, gridspec_kw=None, **fig_kw
) -> tuple[figure.Figure, Union[list[axes.Axes], axes.Axes]]:
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

    if figsize is None:
        figsize = (6.4 * ncols, 4.8 * nrows)
    if gridspec_kw is None:
        gridspec_kw = {"hspace": 0}
    elif gridspec_kw.get("hspace") is None:
        gridspec_kw["hspace"] = 0

    return plt.subplots(
            nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey,
            gridspec_kw=gridspec_kw, subplot_kw=subplot_kw, **fig_kw
    )


def save_fig(
        path: str = '', fig: figure.Figure = None, show: bool = False, **kwargs
) -> None:
    """ Saves the current figure at path

    Parameters
    ----------
    path: str, path to save the figure
    fig: figure.Figure: the figure to save
    show: bool,  show figure, no saving, False: save and show figure
    kwargs any Keywords for Figure.savefig
    """
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path, bbox_inches='tight', **kwargs)
    fig.canvas.draw_idle()
    if show:
        plt.show()
    plt.close(fig)


def calc_specific_freq(r, q, n=1):
    spec_frec = np.reciprocal(np.power(r * q, np.reciprocal(n)))
    spec_frec *= np.reciprocal(2 * np.pi)
    return spec_frec


def predict_circle(r, q, n=1, w=np.logspace(-2, 10, 200)) -> np.array:
    return np.reciprocal(1 / r + q * w ** n * np.exp(np.pi / 2 * n * 1j))


def predict_warburg(a, w=np.logspace(-2, 10, 200)) -> np.array:
    x = np.sqrt(2 * np.pi * w)
    return a * np.reciprocal(x) * (1 - 1j)


def predict_warburg_open(a, b, w=np.logspace(-2, 10, 200)) -> np.array:
    x = np.sqrt(1j * w * b)
    return a * np.reciprocal(x * np.tanh(x))


def predict_warburg_shot(a, b, w=np.logspace(-2, 10, 200)) -> np.array:
    x = np.sqrt(1j * w * b)
    return a * np.reciprocal(x) * np.tanh(x)


def set_plot_params() -> None:
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 10
    rcParams['axes.linewidth'] = 1.1
    rcParams['axes.labelpad'] = 10.0
    plot_color_cycle = cycler(
            'color',
            ['000000', '0000FE', 'FE0000', '008001', 'FD8000', '8c564b',
             'e377c2', '7f7f7f', 'bcbd22', '17becf', ]
    )
    rcParams['axes.prop_cycle'] = plot_color_cycle
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0
    rcParams.update(
            {"figure.subplot.hspace": 0, "figure.subplot.left": 0.11,
             "figure.subplot.right": 0.946, "figure.subplot.bottom": 0.156,
             "figure.subplot.top": 0.965, "xtick.major.size": 4,
             "xtick.minor.size": 2.5, "xtick.major.width": 1.1,
             "xtick.minor.width": 1.1, "xtick.major.pad": 5,
             "xtick.minor.visible": True, "xtick.direction": 'in',
             "xtick.top": True, "ytick.major.size": 4, "ytick.minor.size": 2.5,
             "ytick.major.width": 1.1, "ytick.minor.width": 1.1,
             "ytick.major.pad": 5, "ytick.minor.visible": True,
             "ytick.direction": 'in', "ytick.right": True,
             "lines.markersize": 10, "lines.markeredgewidth": 0.8, }
    )


def _plot_legend(
        ax: axes.Axes = None, loc='upper left', fontsize='small',
        frameon=False, markerscale=0.5, handletextpad=0.1, mode='expand',
        **kwargs
) -> legend.Legend:
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
            loc=loc, fontsize=fontsize, frameon=frameon,
            markerscale=markerscale, handletextpad=handletextpad, mode=mode,
            **kwargs
    )
    return leg


def load_csv_to_df(path: str, sep='\t'):
    return pd.read_csv(path, sep=sep, encoding='unicode_escape')


def load_data(path: str, data_param: list[str] = None) -> list['EISFrame']:
    """
        TODO: WIP
    """
    if data_param is None:
        data_param = ["time/s", "<Ewe>/V", "freq/Hz", "Re(Z)/Ohm",
                      "-Im(Z)/Ohm"]

    if ".csv" in path or ".txt" in path:
        data = load_csv_to_df(path)
    else:
        data = ecf.to_df(path)

    if data.empty:
        warnings.warn(
                "Not valid data file since column cycle number is missing"
        )
        warnings.warn("File location: " + path)

    # check if cycle number is a data column
    if "cycle number" not in data:
        warnings.warn(
                "Not valid data file since column cycle number is missing"
        )
        warnings.warn("File location: " + path)
        return []

    # check if all the parameters are available
    for param in data_param:
        if param not in data:
            warnings.warn(
                    f"Not valid data file since column {param} is missing"
            )
            warnings.warn(f"Availible parameters are {data.columns}")
            warnings.warn("File location: " + path)
            return []

    cycles = []
    for i in range(1, int(max(data['cycle number']))):
        cycle = data[data['cycle number'] == i].reset_index()
        cycles.append(EISFrame(cycle[data_param]))

    # TODO: Sort MB by technique
    # cycles = [mb[mb['cycle number'] == i] for i in range(int(max(mb['cycle
    # number'])))]
    # sequences = []
    # for cycle in cycles:
    #     sequences.append([cycle[cycle['Ns'] == i] for i in range(int(max(
    #     cycle['Ns'])))])
    # TODO: Catch errors if columns not available
    return cycles
