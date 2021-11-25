"""
    This Module has been made for automated plotting of EIS data and fitting.
    The main object is called EISFrame which includes most of the features
    Author: Ueli Sauter
    Date last edited: 25.10.2021
    Python Version: 3.9.7
"""
import logging
import os
import re
import warnings
from typing import Union

import eclabfiles as ecf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from matplotlib import rcParams, cycler, axes, figure, legend
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import minimize, fminbound

from eisplottingtool.parser import parse_circuit


class MarkPoint:
    """ Special point to mark in an eis plot.

    A mark point is given by a specific frequency. The mark point is described
    by a color and a name. A frequency range can be given to narrow the search
    area in frequency space for a data point.
    """

    def __init__(
            self,
            name: str,
            color: str,
            freq: float,
            delta_f: float = -1
            ) -> None:
        """
        Special point in the EIS spectrum

        Parameters
        ----------
        name : str
            Name of the mark point
        color : str
            Color of the mark point
        freq : float
         Specific frequency of the feature
        delta_f : float
            interval to look for datapoints, defualt is 10% of freq
        """
        self.name = name
        self.color = color
        self.freq = freq
        if delta_f <= 0:
            self.delta_f = freq // 10
        else:
            self.delta_f = delta_f
        self.index = -1  # index of the first found data point matching in

        self.left = self.freq - self.delta_f  # left border of freq range
        self.right = self.freq + self.delta_f  # right border of freq range

        self.magnitude = np.floor(np.log10(freq))  # magnitude of the frequency

    def __str__(self):
        out = f"{self.name} @ {self.freq} (1e{self.magnitude}), "
        out += f"color {self.color} "
        out += f"with index {self.index}"
        return out

    def __repr__(self):
        return self.__str__()

    def label(self, freq=None):
        ureg = pint.UnitRegistry()
        if freq is None:
            f = self.freq
        else:
            f = freq[self.index]
        label = f * ureg.Hz
        return f"{label.to_compact():~.0f}"


#  Some default mark points
grain_boundaries = MarkPoint('LLZO-GB', 'blue', freq=3e5, delta_f=5e4)
hllzo = MarkPoint('HLLZO', 'orange', freq=3e4, delta_f=5e3)
lxlzo = MarkPoint('LxLZO', 'lime', freq=2e3, delta_f=5e2)
interface = MarkPoint('Interphase', 'magenta', freq=50, delta_f=5)
ecr_tail = MarkPoint('ECR', 'darkgreen', freq=0.5, delta_f=1)


class Cell:
    """
        Save the characteristics of a cell. Usefull for further calculation.
    """

    def __init__(self, diameter_mm, thickness_mm):
        """
         Initializer of a cell

        Parameters
        ----------
        diameter_mm : float
            diameter of the cell in mm
        thickness_mm height : float
            thickness of the cell in mm
        """
        self.diameter_mm = diameter_mm
        self.height_mm = thickness_mm

        self.area_mm2 = (diameter_mm / 2) ** 2 * np.pi  # area of the cell
        self.volume_mm3 = self.area_mm2 * thickness_mm  # volume of the cell

    def __repr__(self):
        return f"dia: {self.diameter_mm}, area: {self.area_mm2}"

    def __str__(self):
        return "Cell with " + self.__repr__()


class EISFrame:
    """
        EISFrame used to store the data and plot/fit the data.
    """

    def __init__(self, df: pd.DataFrame, params=None) -> None:
        """ Initialises an EISFrame

        An EIS frame can plot a Nyquist plot and a lifecycle lifecycle plot
        of the given cycling data with different default settings.

        Parameters
        ----------
        df : pd.DataFrame

        params : dict
        """
        if params is None:
            params = {}
        self._params = params
        self.df = df
        self._default_mark_points = [grain_boundaries, hllzo, lxlzo, interface,
                                     ecr_tail]
        self.mark_points = self._default_mark_points
        self.lines = {}

    @property
    def time(self) -> np.array:
        return self.df[self._params['time']].values

    @time.setter
    def time(self, value):
        self.df[self._params['time']] = value

    @property
    def impedance(self) -> np.array:
        value = self.df[self._params['real']].values
        value += -1j * self.df[self._params['real']].values
        return value

    @property
    def real(self) -> np.array:
        return self.df[self._params['real']].values

    @property
    def imag(self) -> np.array:
        return self.df[self._params['imag']].values

    @property
    def frequency(self) -> np.array:
        return self.df[self._params['frequency']].values

    @property
    def current(self) -> np.array:
        return self.df[self._params['current']].values

    @property
    def voltage(self) -> np.array:
        return self.df[self._params['voltage']].values

    @property
    def cycle(self) -> np.array:
        return self.df['cycle'].values()

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()

    def __getitem__(self, item):
        if isinstance(item, int):
            df = self.df[self.df[self._params['cycle']] == item].reset_index()
            subframe = self._create_subframe(df)
            return subframe
        return self.df.__getitem__(item)

    def __iter__(self):
        return self.cycle.__iter__()

    def __next__(self):
        return self.cycle.__next__()

    def __len__(self):
        return len(self.df)

    def reset_markpoints(self) -> None:
        """ Resets list markpoints to default

        The default values of the list corresponds to the markpoints for
        grain boundaries, hllzo, lzo, interfacial resistance and ECR.
        """
        self.mark_points = self._default_mark_points

    def plot_nyquist(
            self,
            ax: axes.Axes = None,
            image: str = '',
            cell: Cell = None,
            exclude_start: int = None,
            exclude_end: int = None,
            ls='None',
            marker='o',
            plot_range=None,
            label=None,
            size=12,
            scale=1.5
            ):
        """ Plots a Nyquist plot with the internal dataframe

        Plots a Nyquist plot with the internal dataframe. Will also mark the
        different markpoints on the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axes to plot to
        image : str
             path to image to include in plot
        cell
        exclude_start
        exclude_end
        ls
        marker
        plot_range
        label
        size
        scale

        Returns
        -------
        dictionary
            Contains all the matplotlib.lines.Line2D of the drawn plots
        """
        # label for the plot
        x_label = r"Re(Z)/$\Omega$"
        y_label = r"-Im(Z)/$\Omega$"
        # only look at measurements with frequency data
        mask = self.frequency != 0

        # check if any axes is given, if not GetCurrentAxis from matplotlib
        if ax is None:
            ax = plt.gca()

        # get the x,y data for plotting
        x_data = self.real[mask][exclude_start:exclude_end]
        y_data = self.imag[mask][exclude_start:exclude_end]
        frequency = self.frequency[mask][exclude_start:exclude_end]

        #  # remove all data points with (0,0) and adjust dataframe
        # df = self.df[self.df["Re(Z)/Ohm"] != 0].copy()
        # df = df.reset_index()[exclude_start:exclude_end]

        # # get the x,y data for plotting
        # x_data = df["Re(Z)/Ohm"]
        # y_data = df["-Im(Z)/Ohm"]
        # frequency = df["freq/Hz"]

        # adjust impedance if a cell is given
        if cell is not None:
            x_data = x_data * cell.area_mm2 * 1e-2
            x_label = r"Re(Z)/$\Omega$.cm$^2$"

            y_data = y_data * cell.area_mm2 * 1e-2
            y_label = r"-Im(Z)/$\Omega$.cm$^2$"

        # find indices of mark points. Take first point in freq range
        for mark in self.mark_points:
            # mark.index = -1
            subsequent = (idx for idx, freq in enumerate(frequency) if
                          mark.left < freq < mark.right)
            mark.index = next(subsequent, -1)

        # plot the data
        line = ax.plot(
                x_data,
                y_data,
                marker=marker,
                ls=ls,
                label=label,
                markersize=size,
                )
        lines = {"Data": line}  # store all the lines inside lines

        # plot each mark point with corresponding color and name
        for mark in self.mark_points:
            if mark.index < 0:
                continue
            mark_label = f"{mark.name} @ {mark.label(frequency)}"
            line = ax.plot(
                    x_data[mark.index],
                    y_data[mark.index],
                    marker='o',
                    markerfacecolor=mark.color,
                    markeredgecolor=mark.color,
                    markersize=scale * size,
                    ls='none',
                    label=mark_label
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
        ax.locator_params(nbins=4, prune='upper')
        ax.set_aspect('equal')
        if len(self.mark_points) != 0 or label is not None:
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
            self,
            ax: axes.Axes,
            fit_circuit: str = None,
            fit_guess: list[float] = None,
            fit_bounds: dict[str, tuple] = None,
            path=None,
            cell: Cell = None,
            draw_circle: bool = True,
            draw_circuit: bool = False,
            fit_values=None
            ) -> tuple[list, dict]:
        """
        Fitting for the nyquist TODO: add all options to the function

        Parameters
        ----------
        ax : matplotlib.axes.Axes
             axes to draw the fit to
        fit_circuit : str
            equivalence circuit for the fitting
        fit_guess : list[float]
            initial values for the fitting
        fit_bounds : dict[tuple]
        fit_values
        path : str
        cell : Cell
        draw_circle : bool
            if the corresponding circles should be drawn or not
        draw_circuit : bool
            WIP

        Returns
        -------
        tuple: a tuple containing:
            - parameters list: Fitting parameters with error
            - d dict: dictionary containing all the plots

        """
        # load and prepare data

        frequencies = self.frequency
        z = self.real - 1j * self.imag
        frequencies = np.array(frequencies[np.imag(z) < 0])
        z = np.array(z[np.imag(z) < 0])
        # only for testing purposes like this
        if fit_guess is None:
            fit_guess = [.01, .01, 100, .01, .05, 100, 1]
        if fit_circuit is None:
            fit_circuit = 'R0-p(R1,C1)-p(R2-Wo1,C2)'

        param_info, circ_calc = parse_circuit(fit_circuit)
        param_names = [info[0] for info in param_info]
        # bounds for the fitting
        bounds = []
        if fit_bounds is None:
            fit_bounds = {}

        if isinstance(fit_bounds, dict):
            for i, name in enumerate(param_names):
                if b := fit_bounds.get(name) is not None:
                    bounds.append(b)
                else:
                    # TODO: Get default bounds
                    bounds.append(param_info[i][1])

        # calculate rmse
        def rmse(y_predicted, y_actual, weight=1):
            """ Calculates the root mean squared error between two vectors """
            e = np.abs(np.subtract(y_actual, y_predicted))
            se = np.square(e)
            wse = se / weight
            mse = np.nansum(wse)
            return np.sqrt(mse)

        # prepare optimizing function:
        def opt_func(x):
            params = dict(zip(param_names, x))
            predict = circ_calc(params, frequencies)
            err = rmse(predict, z, np.abs(predict))
            return err

        print(param_info)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in true_divide"
                    )
            warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in true_divide"
                    )
            warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in tanh"
                    )
            warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in double_scalars"
                    )
            warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in power"
                    )
            if fit_values is None:
                opt_result = minimize(
                        opt_func,
                        np.array(fit_guess),
                        bounds=bounds,
                        tol=1e-13,
                        options={'maxiter': 1e4, 'ftol': 1e-9},
                        method='Nelder-Mead'
                        )
                param_values = opt_result.x
            else:
                param_values = np.array(fit_values)

        # print the fitting parameters to the console
        parameters = dict(zip(param_names, param_values))
        print(f"Parameters: {parameters}")

        if path is not None:
            with open(path, 'w') as f:
                f.write(f"parameter, value\n")
                for key in parameters.keys():
                    f.write(f"{key}, {parameters[key]}\n")

        f_pred = np.logspace(-9, 9, 400)
        # plot the fitting result
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in true_divide"
                    )
            warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in true_divide"
                    )
            warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in tanh"
                    )
            warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in double_scalars"
                    )
            warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in power"
                    )
            custom_circuit_fit = circ_calc(parameters, f_pred)

        # adjust impedance if a cell is given
        if cell is not None:
            custom_circuit_fit = custom_circuit_fit * cell.area_mm2 * 1e-2

        line = ax.plot(
                np.real(custom_circuit_fit),
                -np.imag(custom_circuit_fit),
                label="fit",
                color="red",
                zorder=5
                )

        self.lines["fit"] = line
        # check if circle needs to be drawn
        if draw_circle:
            self._plot_semis(fit_circuit, param_info, parameters, cell, ax)

        if draw_circuit:
            self._plot_circuit(fit_circuit, ax)

        _plot_legend(ax)
        return line, parameters

    def plot_lifecycle(
            self,
            ax: axes.Axes = None,
            plot_xrange=None,
            plot_yrange=None,
            label=None,
            ls='-',
            nbinsx=6,
            nbinsy=4,
            **plotkwargs
            ):
        if not {"time/s", "Ewe/V"}.issubset(self.df.columns):
            warnings.warn('Wrong data for a lifecycle Plot', RuntimeWarning)
            return

        # label for the plot
        x_label = r"Time/h"
        y_label = r"$E_{we}$/V"

        # check if any axes is given, if not GetCurrentAxis from matplotlib
        if ax is None:
            ax = plt.gca()

        # remove all data points with (0,0)
        mask = self.voltage != 0
        # df = df[df["Ewe/V"] > 0.05].reset_index()

        x_data = self.time[mask] / 60.0 / 60.0
        y_data = self.voltage[mask]

        line = ax.plot(
                x_data,
                y_data,
                ls=ls,
                label=label,
                color='black',
                **plotkwargs
                )

        # additional configuration for the plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plot_xrange is None:
            xrange = max(*x_data * 1.05, *ax.get_xlim())
            ax.set_xlim(-xrange * 0.01, xrange)
        else:
            ax.set_xlim(*plot_xrange)

        if plot_yrange is None:
            limits = [np.median(y_data) * 2, *ax.get_ylim()]
            print(limits)
            yrange = max(limits)
            ax.set_ylim(-yrange, yrange)
        else:
            ax.set_ylim(*plot_yrange)

        ax.locator_params(axis='x', nbins=nbinsx)
        ax.locator_params(axis='y', nbins=nbinsy, prune='both')

        if label is not None:
            _plot_legend(ax)

        return line

    def _plot_semis(
            self,
            circuit: str,
            param_info: list,
            param_values: dict,
            cell=None,
            ax: axes.Axes = None
            ):
        """
        plots the semicircles to the corresponding circuit elements.

        the permitted elements are p(R,CPE), p(R,C) or
        TODO: any Warburg element
        @param circuit:
        @param ax:
        Parameters
        ----------
        circuit : CustomCircuit
            CustomCircuit
        ax : matplotlib.axes.Axes
             axes to be plotted to

        Returns
        -------
        nothing at the moment
        """
        # check if axes is given, else get current axes
        if ax is None:
            ax = plt.gca()

        elem_infos = []

        # split the circuit in to elements connected through series
        elements = re.split(r"-(?![^\(]*\))", circuit)
        spec_frequencies = []
        for e in elements:
            elem_info, elem_eval = parse_circuit(e)
            elem_names = [info[0] for info in elem_info]

            if match := re.match(r'(?=.*(R_?\d?))(?=.*(C(?:PE)?_?\d?))', e):
                res = param_values.get(match.group(1))
                cap = [param_values.get(key) for key in param_values if
                       match.group(2) in key]

                def calc_specific_freq(r, c, n=1):
                    return 1.0 / (r * c) ** n / 2 / np.pi

                specific_frequency = calc_specific_freq(res, *cap)

                # max_x = fminbound(
                #         lambda x: np.imag(elem_eval(param_values, x)),
                #         1,
                #         1e12
                #         )

            elif match := re.match(r'(W[os]?_?\d?)', e):
                war = [param_values.get(match.group(1))]
                if len(war) == 2:
                    specific_frequency = 1.0 / war[1]
                else:
                    specific_frequency = 1e-2
            else:
                continue

            freq = np.logspace(-9, 9, 180)
            elem_impedance = elem_eval(param_values, freq)

            if cell is not None:
                elem_impedance = elem_impedance * cell.area_mm2 * 1e-2

            elem_infos.append((elem_impedance, specific_frequency))
            # get previous resistors
        color = 'black'

        elem_infos.sort(key=lambda x: x[1], reverse=True)
        # check with which mark point the circle is associated by
        # comparing magnitudes
        prev_imp = 0
        for index, elem_info in enumerate(elem_infos):
            elem_impedance = elem_info[0]
            elem_spec_freq = elem_info[1]
            color = 'black'
            specific_freq_magnitude = np.floor(np.log10(elem_spec_freq))
            print(10 * "*")
            print(specific_freq_magnitude)
            for mark in self.mark_points:
                print(mark.name, mark.magnitude)
                if specific_freq_magnitude == mark.magnitude:
                    color = mark.color
                    break
                if specific_freq_magnitude <= 0:
                    color = min(
                            self.mark_points, key=lambda x: x.magnitude
                            ).color
                    break

            # draw circle
            if cell is not None:
                elem_impedance = elem_impedance * cell.area_mm2 * 1e-2

            ax.fill_between(
                    np.real(elem_impedance) + prev_imp,
                    -np.imag(elem_impedance),
                    color=color,
                    alpha=0.5,
                    zorder=0,
                    ls='None'
                    )
            print(prev_imp)
            prev_imp += np.real(elem_impedance)[0]

        return

    def _plot_circuit(self, circuit: str, ax: axes.Axes = None):
        # TODO: Look at SchemDraw to draw circuit and color with different
        #  mark  points

        # check if axes is given, else get current axes
        if ax is None:
            ax = plt.gca()

        # read out details about circuit
        circuit_string = circuit.circuit
        names = circuit.get_param_names()[0]

        elements = circuit_string.split('-')

        marks = self.mark_points

        ax.plot(np.linspace(5))
        # count how many components in string -> n
        # if n%2  == 1 -> draw one element in center and do for (n-1) i.e.
        # even n
        # if n%2 == 0 -> draw elements moved by 0.5 (maybe 0.25?) up and down
        return

    def _create_subframe(self, df):
        subframe = EISFrame(df)
        subframe.mark_points = self.mark_points
        subframe._params = self._params
        return subframe


def create_fig(
        nrows: int = 1,
        ncols: int = 1,
        sharex='all',
        sharey='all',
        figsize=None,
        subplot_kw=None,
        gridspec_kw=None,
        top_ticks=False,
        **fig_kw
        ) -> tuple[figure.Figure, Union[axes.Axes, list[axes.Axes]]]:
    """ Creates the figure, axes for the plots and set the style of the plot

    Parameters
    ----------
    nrows : int
        number of rows
    ncols :
        number of columns
    sharex
    sharey
    figsize
    subplot_kw
    gridspec_kw
    top_ticks
    fig_kw

    Returns
    -------
    the figure and list of created axes
    """
    set_plot_params()

    if figsize is None:
        figsize = (6.4 * ncols, 4.8 * nrows)
    if gridspec_kw is None:
        gridspec_kw = {"hspace": 0}
    elif gridspec_kw.get("hspace") is None:
        gridspec_kw["hspace"] = 0

    fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw=gridspec_kw,
            subplot_kw=subplot_kw,
            **fig_kw
            )

    if top_ticks:
        axs[0].xaxis.set_tick_params(which="both", labeltop=True)

    return fig, axs


def save_fig(
        path: str = '', fig: figure.Figure = None, show: bool = False, **kwargs
        ) -> None:
    """ Saves the current figure at path

    Parameters
    ----------
    path : str
        path to save the figure
    fig : matplotlib.figure.Figure
        the figure to save
    show : bool
        show figure, no saving, False: save and show figure
    **kwargs
        any Keywords for Figure.savefig
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


def flat2gen(alist):
    for item in alist:
        if isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item


def set_plot_params() -> None:
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 22
    rcParams['axes.linewidth'] = 1.1
    rcParams['axes.labelpad'] = 4.0
    plot_color_cycle = cycler(
            'color',
            ['000000', '0000FE', 'FE0000', '008001', 'FD8000', '8c564b',
             'e377c2', '7f7f7f', 'bcbd22', '17becf', ]
            )
    rcParams['axes.prop_cycle'] = plot_color_cycle
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0
    rcParams.update(
            {
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
                }
            )


def _plot_legend(
        ax: axes.Axes = None,
        loc='upper left',
        fontsize='small',
        frameon=False,
        markerscale=1,
        handletextpad=0.1,
        mode='expand',
        **kwargs
        ) -> legend.Legend:
    """ Adds legend to an axes

    Parameters
    ----------
    ax
    loc
    fontsize
    frameon
    markerscale
    handletextpad
    mode
    kwargs

    Returns
    -------

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


def load_csv_to_df(path: str, sep='\t'):
    return pd.read_csv(path, sep=sep, encoding='unicode_escape')


def _get_default_data_param(columns):
    params = {}
    for col in columns:
        if match := re.match(r'Ewe[^|]*', col):
            params['voltage'] = match.group()
        elif match := re.match(r'I/mA[^|]*', col):
            params['current'] = match.group()
        elif match := re.match(r'Re\(Z(we-ce)?\)[^|]*', col):
            params['real'] = match.group()
        elif match := re.match(r'-Im\(Z(we-ce)?\)[^|]*', col):
            params['imag'] = match.group()
        elif match := re.match(r'time[^|]*', col):
            params['time'] = match.group()
        elif match := re.match(r'(z )?cycle( number)?[^|]*', col):
            params['cycle'] = match.group()
        elif match := re.match(r'freq[^|]*', col):
            params['frequency'] = match.group()
    return params


def load_data(
        path: str,
        file: Union[str, list[str]] = None,
        sep='\t',
        cont_time: bool = True,
        data_param: dict = None,
        ) -> Union[EISFrame, list[EISFrame]]:
    """
        loads the data from the given path into an EISFrame
    """
    if isinstance(file, list):
        end_time = 0
        data = []
        for f in file:
            d = load_data(path + f, sep=sep, data_param=data_param)
            if cont_time:
                start_time = d[0].time.iloc[0]
                for c in d:
                    c.time += end_time - start_time
                end_time = d[-1].time.iloc[-1]

            data += d

        return data

    if file is not None:
        path = path + file

    __, ext = os.path.splitext(path)

    if ".csv" in path or ".txt" in ext:
        data = load_csv_to_df(path, sep)
    elif ext == '.mpr' or ext == '.mpt':
        data = ecf.to_df(path)
    else:
        warnings.warn("Datatype not supported")
        return []

    # check if data was loaded
    if data.empty:
        warnings.warn("No data was loaded")
        logging.info("File location: " + path)
        return []

    # check if all the parameters are available
    if data_param is None:
        data_param = _get_default_data_param(data.columns)
    else:
        for param in data_param:
            if param not in data:
                warnings.warn(
                        f"Not valid data file since column {param} is missing"
                        )
                print(f"Availible parameters are {data.columns}")
                logging.info("File location: " + path)
                return []

    if (cycle_param := data_param.get('cycle')) is None:
        print("No cycles detected")
        return EISFrame(data, params=data_param)

    cycles = []

    max_cyclenumber = int(max(data[cycle_param]))
    min_cyclenumber = int(min(data[cycle_param]))

    for i in range(min_cyclenumber, max_cyclenumber + 1):
        cycle = data[data[cycle_param] == i].reset_index()
        cycles.append(EISFrame(cycle, params=data_param))
    return cycles
