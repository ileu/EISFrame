"""
    This Module has been made for automated plotting of EIS data and fitting.
    The main object is called EISFrame which includes most of the features
    Author: Ueli Sauter
    Date last edited: 25.10.2021
    Python Version: 3.9.7
"""
import json
import logging
import os
import re
import warnings
from typing import TypeVar, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from matplotlib import axes
from matplotlib.patches import BoxStyle
from matplotlib.ticker import AutoMinorLocator

from eisplottingtool.parser.CircuitParser import parse_circuit
from eisplottingtool.utils.UtilClass import Cell, default_mark_points
from eisplottingtool.utils.UtilFunctions import plot_legend
from eisplottingtool.utils.fitting import fit_routine

Logger = logging.getLogger(__name__)
T = TypeVar('T', bound='Parent')


class EISFrame:
    """
        EISFrame used to store the data and plot/fit the data.
    """

    def __init__(self, df: pd.DataFrame = None, column_names=None, **kwargs) -> None:
        """ Initialises an EISFrame

        An EIS frame can plot a Nyquist plot and a lifecycle plot
        of the given cycling data with different default settings.

        Parameters
        ----------
        df : pd.DataFrame

        column_names : dict

        **kwargs: dict
            path: str
                Path to data file
            circuit: str
                Equivilant circuit for impedance
            cell: Cell
                Battery cell for normalizing
        """
        if column_names is None:
            column_names = {}
        self.column_names = column_names.copy()

        if "path" in kwargs:
            self.path = kwargs["path"]

        elif df is not None:
            self.df = df
        else:
            raise ValueError()

        if 'real' not in column_names:
            if 'phase' in column_names and 'abs' in column_names:
                self.df['real'] = df[column_names['abs']] * np.cos(
                    df[column_names['phase']] / 360.0 * 2 * np.pi
                )
                self.column_names['real'] = 'real'

        if 'imag' not in column_names:
            if 'phase' in column_names and 'abs' in column_names:
                self.df['imag'] = -df[column_names['abs']] * np.sin(
                    df[column_names['phase']] / 360.0 * 2 * np.pi
                )
                self.column_names['imag'] = 'imag'

        self.mark_points = default_mark_points
        self.lines = {}

        self.eis_params = kwargs

    @property
    def time(self) -> np.array:
        return self.df[self.column_names['time']].values

    @time.setter
    def time(self, value):
        self.df[self.column_names['time']] = value

    @property
    def impedance(self) -> np.array:
        value = self.df[self.column_names['real']].values
        value += -1j * self.df[self.column_names['real']].values
        return value

    @property
    def real(self) -> np.array:
        return self.df[self.column_names['real']].values

    @property
    def imag(self) -> np.array:
        return self.df[self.column_names['imag']].values

    @property
    def frequency(self) -> np.array:
        return self.df[self.column_names['frequency']].values

    @property
    def current(self) -> np.array:
        return self.df[self.column_names['current']].values

    @property
    def voltage(self) -> np.array:
        return self.df[self.column_names['voltage']].values

    @property
    def cycle(self) -> np.array:
        return self.df['cycle'].values()

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()

    def __getitem__(self, item):
        if isinstance(item, int):
            self.eis_params['cycle'] = item
            return self
        return self.df.__getitem__(item)

    def __len__(self):
        return len(self.cycle)

    def reset_markpoints(self) -> None:
        """ Resets list markpoints to default

        The default values of the list corresponds to the markpoints for
        grain boundaries, hllzo, lzo, interfacial resistance and ECR.

        ? depracted ?
        """
        self.mark_points = default_mark_points

    def plot_nyquist(
        self,
        ax: axes.Axes = None,
        image: str = '',
        cell: Cell = None,
        exclude_start: int = None,
        exclude_end: int = None,
        show_freq: bool = False,
        color=None,
        ls='None',
        marker=None,
        plot_range=None,
        label=None,
        size=6,
        scale=1.5,
        normalize=None,
        unit=None,
        show_legend=True,
        show_mark_label=True,
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
        show_freq
        color
        ls
        marker
        plot_range
        label
        size
        scale
        normalize
        unit
        show_legend
        show_mark_label

        Returns
        -------
        dictionary
            Contains all the matplotlib.lines.Line2D of the drawn plots
        """
        # initialize
        if marker is None:
            marker = 'o'

        # label for the plot
        if unit is None:
            if cell is None:
                x_label = r"Re(Z)/$\Omega$"
                y_label = r"-Im(Z)/$\Omega$"
            else:
                x_label = r"Re(Z)/$\Omega$.cm$^2$"
                y_label = r"-Im(Z)/$\Omega$.cm$^2$"
        else:
            x_label = rf"Re(Z) [{unit}]"
            y_label = rf"-Im(Z) [{unit}]"
        # only look at measurements with frequency data
        mask = self.frequency != 0

        # check if any axes is given, if not GetCurrentAxis from matplotlib
        if ax is None:
            ax = plt.gca()

        # get the x,y data for plotting
        x_data = self.real[mask][exclude_start:exclude_end]
        y_data = self.imag[mask][exclude_start:exclude_end]
        frequency = self.frequency[mask][exclude_start:exclude_end]

        # adjust impedance if a cell is given
        if normalize is None:
            if cell is None:
                def normalize(data, c: Cell):
                    return data
            else:
                def normalize(data, c: Cell):
                    return data * c.area_mm2 * 1e-2

        x_data = normalize(x_data, cell)
        y_data = normalize(y_data, cell)

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
            color=color,
            ls=ls,
            label=label,
            markersize=size,
        )
        lines = {"Data": line}  # store all the lines inside lines

        # plot each mark point with corresponding color and name
        for mark in self.mark_points:
            if mark.index < 0:
                continue
            if show_mark_label:
                mark_label = f"{mark.name} @ {mark.label(frequency)}"
            else:
                mark_label = None
            line = ax.plot(
                x_data[mark.index],
                y_data[mark.index],
                marker=marker,
                markerfacecolor=mark.color,
                markeredgecolor=mark.color,
                markersize=scale * size,
                ls='none',
                label=mark_label
            )
            lines[f"MP-{mark.name}"] = line

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
            if show_legend:
                plot_legend(ax)

        # add lines to the axes property
        self.lines.update(lines)

        if show_freq:
            ureg = pint.UnitRegistry()

            lower_freq = frequency[-1] * ureg.Hz
            upper_freq = frequency[0] * ureg.Hz
            lower_label = f"{lower_freq.to_compact():~.0f}"
            upper_label = f"{upper_freq.to_compact():~.0f}"

            ax.text(
                0.99,
                0.99,
                f"Freq. Range:\n {upper_label} - {lower_label}",
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                size='xx-small',
                multialignment='center',
                bbox=dict(
                    facecolor='white',
                    alpha=1.0,
                    boxstyle=BoxStyle("Round", pad=0.2)
                )
            )

        # if a path to an image is given, also plot it
        if image:
            imax = ax.inset_axes([.05, .5, .9, .2])
            img = plt.imread(image)
            imax.imshow(img)
            imax.axis('off')
            lines["Image"] = img
            return lines, imax

        return lines

    def plot_bode(
        self,
        ax: axes.Axes = None,
        cell: Cell = None,
        exclude_start: int = None,
        exclude_end: int = None,
        ls='None',
        marker='o',
        plot_range=None,
        param_values=None,
        param_circuit=None,
        size=12,
    ):
        """ Plots a Nyquist plot with the internal dataframe

        Plots a Nyquist plot with the internal dataframe. Will also mark the
        different markpoints on the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axes to plot to
        cell
        exclude_start
        exclude_end
        ls
        marker
        plot_range
        param_values
        param_circuit
        size

        Returns
        -------
        dictionary
            Contains all the matplotlib.lines.Line2D of the drawn plots
        """
        # label for the plot
        x_label = r"Frequency/log(Hz)"
        y_label = r"Impedance/$\Omega$"
        # only look at measurements with frequency data
        mask = self.frequency != 0

        # check if any axes is given, if not GetCurrentAxis from matplotlib
        if ax is None:
            ax = plt.gca()

        # get the x,y data for plotting
        real_data = self.real[mask][exclude_start:exclude_end]
        imag_data = self.imag[mask][exclude_start:exclude_end]
        frequency = self.frequency[mask][exclude_start:exclude_end]

        #  # remove all data points with (0,0) and adjust dataframe
        # df = self.df[self.df["Re(Z)/Ohm"] != 0].copy()
        # df = df.reset_index()[exclude_start:exclude_end]

        # adjust impedance if a cell is given
        if cell is not None:
            real_data = real_data * cell.area_mm2 * 1e-2
            x_label = r"Re(Z)/$\Omega$.cm$^2$"

            imag_data = imag_data * cell.area_mm2 * 1e-2
            y_label = r"-Im(Z)/$\Omega$.cm$^2$"

        # plot the data
        line_real = ax.semilogx(
            frequency,
            real_data,
            marker=marker,
            ls=ls,
            color='red',
            label="Re(Z)",
            markersize=size,
        )

        line_imag = ax.semilogx(
            frequency,
            imag_data,
            marker=marker,
            ls=ls,
            color='blue',
            label="-Im(Z)",
            markersize=size,
        )

        lines = {
            "Real Data": line_real,
            "Imag Data": line_imag
        }  # store all the lines inside lines

        # additional configuration for the plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plot_range is None:
            ax.set_xlim(-max(frequency) * 0.05, max(frequency) * 1.05)
        else:
            ax.set_xlim(*plot_range)

        ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))

        if param_values and param_circuit:
            param_info, circ_calc = parse_circuit(param_circuit)
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
                custom_circuit_fit = circ_calc(param_values, frequency)
                real_fit = np.real(custom_circuit_fit)
                imag_fit = np.imag(custom_circuit_fit)
                fit_real = ax.semilogx(
                    frequency,
                    real_fit,
                    ls='-',
                    color='black',
                    label="Re(Z)",
                    markersize=size,
                )

                fit_imag = ax.semilogx(
                    frequency,
                    -imag_fit,
                    ls='-',
                    color='black',
                    label="-Im(Z)",
                    markersize=size,
                )

                lines["Real Fit"] = fit_real
                lines["Imag Fit"] = fit_imag

        plot_legend(ax)

        return lines

    def fit_nyquist(
        self,
        ax: axes.Axes,
        fit_circuit: str = None,
        fit_guess: dict[str, float] = None,
        fit_bounds: dict[str, tuple] = None,
        fit_constants: list[str] = None,
        path=None,
        cell: Cell = None,
        draw_circle: bool = True,
        draw_circuit: bool = False,
        fit_values=None,
        tot_imp=None,
        data_slice=None,
        **kwargs
    ) -> tuple[dict, list]:
        """
        Fitting for the nyquist

        Parameters
        ----------
        ax : matplotlib.axes.Axes
             axes to draw the fit to
        fit_circuit : str
            equivalence circuit for the fitting
        fit_guess : dict[str, float]
            initial values for the fitting
        fit_bounds : dict[str, tuple]
        fit_constants : list[str]
        fit_values
        path : str
        cell : Cell
        draw_circle : bool
            if the corresponding circles should be drawn or not
        draw_circuit : bool
            WIP
        tot_imp
        data_slice

        Returns
        -------
        tuple: a tuple containing:
            - d dict: dictionary containing all the plots
            - parameters list: Fitting parameters with error

        """
        # load and prepare data

        if data_slice is None:
            data_slice = slice(3, None)

        frequencies = self.frequency
        z = self.real - 1j * self.imag
        frequencies = np.array(frequencies[np.imag(z) < 0])[data_slice]
        z = np.array(z[np.imag(z) < 0])[data_slice]

        # check and parse circuit

        if fit_circuit:
            pass
        elif "circuit" in self.eis_params:
            fit_circuit = self.eis_params["circuit"]
        else:
            raise ValueError("No fit circuit given")

        param_info, circ_calc = parse_circuit(fit_circuit)

        # check and prepare parameters

        if fit_guess:
            pass
        elif "fit_guess" in self.eis_params:
            fit_guess = self.eis_params["fit_guess"]
        else:
            raise ValueError("No fit guess given")

        if fit_bounds is None:
            fit_bounds = {}

        if fit_constants is None:
            fit_constants = []

        param_names = []
        param_values = {}
        param_guess = []
        param_bounds = []

        for p in param_info:
            name = p.name
            if name in fit_guess:
                value = fit_guess.get(name)
                param_values[name] = value

                if name in fit_bounds:
                    p.bounds = fit_bounds.get(name)

                if name in fit_constants:
                    p.fixed = True
                    fit_guess.pop(name)
                else:
                    p.fixed = False
                    param_bounds.append(p.bounds)
                    param_guess.append(value)
                    param_names.append(name)
            else:
                raise ValueError(f"No initial value given for {name}")

        # calculate the weight of each datapoint
        def weight(error, value):
            square_value = value.real ** 2 + value.imag ** 2
            return np.true_divide(error, square_value)

        # calculate rmse
        def rmse(y_predicted, y_actual):
            """ Calculates the root mean squared error between two vectors """
            e = np.abs(np.subtract(y_actual, y_predicted))
            se = np.square(e)
            wse = weight(se, y_actual)
            mse = np.nansum(wse)
            return np.sqrt(mse)

        # prepare optimizing function:
        def opt_func(x: list[float]):
            params = dict(zip(param_names, x))
            param_values.update(params)
            predict = circ_calc(param_values, frequencies)
            err = rmse(predict, z)
            return err

        if fit_values:
            param_values = np.array(fit_values)
        else:
            if tot_imp is None:
                opt_result = fit_routine(
                    opt_func,
                    param_guess,
                    param_bounds,
                )
            else:
                # def condition(x, v=False):
                #     params = param_info.get_namevaluepairs()
                #     predict = circ_calc(params, 1e-13)
                #     if v:
                #         print(tot_imp - predict.real)
                #     err = np.abs(tot_imp - predict.real)
                #     return err

                def condition(params):
                    res = params["R2"] + params["Wss1_R"]
                    return rmse(res, tot_imp)

                def opt_func(x):
                    params = dict(zip(param_names, x))
                    param_values.update(params)
                    predict = circ_calc(param_values, frequencies)
                    main_err = rmse(predict, z)
                    last_predict = circ_calc(param_values, 1e-13)
                    cond_err = condition(param_values)
                    err = main_err + cond_err
                    return err

                opt_result = fit_routine(
                    opt_func,
                    param_guess,
                    param_bounds,
                )

            param_values = opt_result.x

        # print the fitting parameters to the console
        report = f"Fitting report:\n"
        report += f"Equivivalent circuit: {fit_circuit}\n"
        report += "Parameters: \n"
        for p_value, p_info in zip(param_values, param_info):
            p_info.value = p_value
            report += f"\t {p_info}\n"

        Logger.info(report)

        if path is not None:
            if not os.path.isdir(os.path.dirname(path)):
                # TODO: look at exist_ok=True
                os.makedirs(os.path.dirname(path))
            with open(path, 'w') as f:
                json.dump(
                    param_info, f, default=lambda o: o.__dict__, indent=1
                )

        lines = self.plot_fit(
            ax,
            fit_circuit,
            param_info,
            data_slice=data_slice,
            cell=cell,
            color='red'
        )
        # check if circle needs to be drawn
        if draw_circle:
            self._plot_semis(fit_circuit, param_info, cell, ax)

        if draw_circuit:
            self._plot_circuit(fit_circuit, ax)

        plot_legend(ax)
        result = {}
        for p in param_info:
            result[p.name] = p.value
        return lines, result

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
            yrange = max(limits)
            ax.set_ylim(-yrange, yrange)
        else:
            ax.set_ylim(*plot_yrange)

        ax.locator_params(axis='x', nbins=nbinsx)
        ax.locator_params(axis='y', nbins=nbinsy, prune='both')

        if label is not None:
            plot_legend(ax)

        return line

    def _plot_semis(
        self,
        circuit: str,
        param_info: list,
        cell=None,
        ax: axes.Axes = None
    ):
        """
        plots the semicircles to the corresponding circuit elements.

        The recognised elements are p(R,CPE), p(R,C) or any Warburg element

        Parameters
        ----------
        circuit : CustomCircuit
            CustomCircuit
        param_info
        cell
        ax : matplotlib.axes.Axes
             axes to be plotted to

        Returns
        -------
        nothing at the moment
        """
        # check if axes is given, else get current axes
        if ax is None:
            ax = plt.gca()

        param_values = {info.name: info.value for info in param_info}
        elem_infos = []

        # split the circuit in to elements connected through series
        elements = re.split(r"-(?![^(]*\))", circuit)
        for e in elements:
            elem_info, elem_eval = parse_circuit(e)

            if match := re.match(r'(?=.*(R_?\d?))(?=.*(C(?:PE)?_?\d?))', e):
                res = param_values.get(match.group(1))
                cap = [param_values.get(key) for key in param_values if
                       match.group(2) in key]

                def calc_specific_freq(r, c, n=1):
                    return 1.0 / (r * c) ** n / 2 / np.pi

                specific_frequency = calc_specific_freq(res, *cap)

            elif match := re.match(r'(W[os]?_?\d?)', e):
                war = [param_values.get(key) for key in param_values if
                       match.group(1) in key]
                if len(war) == 2:
                    specific_frequency = 1.0 / war[1]
                else:
                    specific_frequency = 1e-2
            elif re.match(r'(R_?\d?)', e):
                specific_frequency = 1e20
            else:
                continue

            freq = np.logspace(-9, 9, 180)
            elem_impedance = elem_eval(param_values, freq)

            if cell is not None:
                elem_impedance = elem_impedance * cell.area_mm2 * 1e-2

            elem_infos.append((elem_impedance, specific_frequency))

        elem_infos.sort(key=lambda x: x[1], reverse=True)
        # check with which mark point the circle is associated by
        # comparing magnitudes
        prev_imp = 0
        for index, elem_info in enumerate(elem_infos):
            elem_impedance = elem_info[0]
            elem_spec_freq = elem_info[1]
            specific_freq_magnitude = np.floor(np.log10(elem_spec_freq))
            if specific_freq_magnitude <= 0:
                color = min(
                    self.mark_points, key=lambda x: x.magnitude
                ).color
            else:
                for mark in self.mark_points:
                    if specific_freq_magnitude == mark.magnitude:
                        color = mark.color
                        break
                else:
                    prev_imp += np.real(elem_impedance)[0]
                    continue

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
            prev_imp += np.real(elem_impedance)[0]

        return

    def _plot_circuit(self, circuit: str, ax: axes.Axes = None):
        # TODO: Look at SchemDraw to draw circuit and color with different
        #  mark points
        pass

    @classmethod
    def from_file(cls: Type[T], path) -> T:
        # TODO: create EISFrame from file location
        pass
