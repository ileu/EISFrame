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
from typing import TypeVar

import eclabfiles as ecf
import numpy as np
import pandas as pd
import pint
from matplotlib import axes, pyplot as plt
from matplotlib.patches import BoxStyle
from matplotlib.ticker import AutoMinorLocator

from .parser import parse_circuit
from .utils import plot_legend
from .utils.UtilClass import Cell, default_mark_points
from .utils.fitting import fit_routine

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound="Parent")

control_types = {0: "CC", 3: "CR", 4: "Rest", 8: "PEIS"}


def load_df_from_path(path):
    return None


def _get_default_data_param(columns):
    col_names = {}
    for col in columns:
        if match := re.match(r"Ewe[^|]*", col):
            col_names["voltage"] = match.group()
        elif match := re.match(r"I[^|]*", col):
            col_names["current"] = match.group()
        elif match := re.match(r"Re\(Z(we-ce)?\)[^|]*", col):
            col_names["real"] = match.group()
        elif match := re.match(r"-Im\(Z(we-ce)?\)[^|]*", col):
            col_names["imag"] = match.group()
        elif match := re.match(r"Phase\(Z(we-ce)?\)[^|]*", col):
            col_names["phase"] = match.group()
        elif match := re.match(r"\|Z(we-ce)?\|[^|]*", col):
            col_names["abs"] = match.group()
        elif match := re.match(r"time[^|]*", col):
            col_names["time"] = match.group()
        elif match := re.match(r"(z )?cycle( number)?[^|]*", col):
            col_names["cycle"] = match.group()
        elif match := re.match(r"freq[^|]*", col):
            col_names["frequency"] = match.group()
        elif match := re.match(r"Ns", col):
            col_names["Ns"] = match.group()
    return col_names


class EISFrame:
    """
    EISFrame stores EIS data and plots/fits the data.
    """

    def __init__(
        self, name: str = None, path: str = None, df: pd.DataFrame = None, **kwargs
    ) -> None:
        """Initialises an EISFrame

        An EIS frame can plot a Nyquist plot and a lifecycle plot
        of the given cycling data with different default settings.

        Parameters
        ----------
        name: str

        path: str
            Path to data file

        df : pd.DataFrame

        **kwargs: dict
            circuit: str
                Equivilant circuit for impedance
            cell: Cell
                Battery cell for normalizing
        """
        self.eis_params = kwargs
        self.df = df
        self.mark_points = default_mark_points
        self.eis_params["Lines"] = {}

        if name is not None:
            self.eis_params["name"] = name

        if path is not None:
            self.eis_params["path"] = path

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()

    def __getitem__(self, item):
        self.eis_params["selection"] = item
        if self.df is None:
            self.load()
        return EISFrame(df=self.select_data(item), **self.eis_params)

    @property
    def time(self) -> np.array:
        if self.df is None:
            self.load()
        return self.df[self.eis_params["time"]].values

    @property
    def impedance(self) -> np.array:
        if self.df is None:
            self.load()
        value = self.df[self.eis_params["real"]].values
        value += -1j * self.df[self.eis_params["real"]].values
        return value

    @property
    def real(self) -> np.array:
        if self.df is None:
            self.load()
        return self.df[self.eis_params["real"]].values

    @property
    def imag(self) -> np.array:
        if self.df is None:
            self.load()
        return self.df[self.eis_params["imag"]].values

    @property
    def frequency(self) -> np.array:
        if self.df is None:
            self.load()
        return self.df[self.eis_params["frequency"]].values

    @property
    def current(self) -> np.array:
        if self.df is None:
            self.load()
        return self.df[self.eis_params["current"]].values

    @property
    def voltage(self) -> np.array:
        if self.df is None:
            self.load()
        return self.df[self.eis_params["voltage"]].values

    def select_data(self, selection):
        if isinstance(selection, int):
            return self.df.loc[selection]
        elif isinstance(selection, tuple):
            return self.df.loc[selection]
        elif isinstance(selection, dict):
            cyc = selection.get("cycle")
            ns = selection.get("sequence")
            if ns and cyc:
                return self.df.loc[(cyc, ns)]
            elif ns:
                return self.df.loc[(slice(None, ns))]
            elif cyc:
                return self.df.loc[cyc]
        elif isinstance(selection, str):
            if selection in self.eis_params:
                return self.eis_params[selection]
        else:
            raise ValueError("Invalid Selection")

    def load(self, path=None):
        # TODO: combine two or more files together
        if path is None:
            path = self.eis_params["path"]

        ext = os.path.splitext(path)[1][1:]

        if ext in {"csv", "txt"}:
            data = pd.read_csv(path, sep=",", encoding="unicode_escape")
        elif ext in {"mpr", "mpt"}:
            data = ecf.to_df(path)
        else:
            raise ValueError(f"Datatype {ext} not supported")

        if data.empty:
            raise ValueError(f"File {path} has no data")

        col_names = _get_default_data_param(data.columns)
        if "cycle" not in col_names:
            raise ValueError(f"No cycles detetected in file {path}.")

        if "Ns" in col_names:
            data.set_index([col_names["cycle"], col_names.get("Ns")], inplace=True)
        else:
            data.set_index([col_names["cycle"]], inplace=True)

        ctrl = []
        for p in data.attrs["params"]:
            ctrl.append(p["ctrl_type"])

        self.df = data.sort_index()
        self.eis_params.update(col_names)

    def modify_data(self):
        # TODO
        pass

    def plot_nyquist(
        self,
        ax: axes.Axes = None,
        cycle=None,
        exclude_data=None,
        show_freq: bool = False,
        color=None,
        ls="None",
        marker=None,
        plot_range=None,
        show_legend=True,
        show_mark_label=True,
        **kwargs,
    ):
        """Plots a Nyquist plot with the internal dataframe

        Plots a Nyquist plot with the internal dataframe. Will also mark the
        different markpoints on the plot.

        https://stackoverflow.com/questions/62308183/wrapper-function-for-matplotlib-pyplot-plot
        https://codereview.stackexchange.com/questions/101345/generic-plotting-wrapper-around-matplotlib

        Parameters
        ----------
        ax
            matplotlib axes to plot to
        cycle
        exclude_data
        show_freq
        color
        ls
        marker
        plot_range
        label
        show_legend
        show_mark_label

        kwargs :
            color?

        Returns
        -------
        dictionary
            Contains all the matplotlib.lines.Line2D of the drawn plots
        """
        if self.df is None:
            self.load()

        # only look at measurements with frequency data
        mask = self.frequency != 0

        if exclude_data is None:
            exclude_data = slice(None)
        # get the x,y data for plotting
        x_data = self.real[mask][exclude_data]
        y_data = self.imag[mask][exclude_data]
        frequency = self.frequency[mask][exclude_data]

        # label for the plot
        if "x_label" not in kwargs:
            x_label = rf"Re(Z) [Ohm]"
        else:
            x_label = kwargs.get("x_label")

        if "y_label" not in kwargs:
            y_label = rf"-Im(Z) [Ohm]"
        else:
            y_label = kwargs.get("y_label")

        if "name" not in kwargs:
            name = self.eis_params.get("name")
        else:
            name = kwargs["name"]

        # check if any axes is given, if not GetCurrentAxis from matplotlib
        if ax is None:
            ax = plt.gca()

        # find indices of mark points. Take first point in freq range
        for mark in self.mark_points:
            for ind, freq in enumerate(frequency):
                if mark.left < freq < mark.right:
                    mark.index = ind
                    break
            else:
                mark.index = -1

        size = 6
        scale = 1.5

        # plot the data
        line = ax.plot(
            x_data,
            y_data,
            marker=marker or self.eis_params.get("marker", "o"),
            color=color,
            ls=ls,
            label=name,
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
                marker=marker if marker else "o",
                markerfacecolor=mark.color,
                markeredgecolor=mark.color,
                markersize=scale * size,
                ls="none",
                label=mark_label,
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
        ax.locator_params(nbins=4, prune="upper")
        ax.set_aspect("equal")
        if not all(ax.get_legend_handles_labels()):
            if show_legend:
                plot_legend(ax)

        # add lines to the axes property
        self.eis_params["Lines"].update(lines)

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
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                size="xx-small",
                multialignment="center",
                bbox=dict(
                    facecolor="white", alpha=1.0, boxstyle=BoxStyle("Round", pad=0.2)
                ),
            )

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
        param_bounds = []

        for p in param_info:
            name = p.name
            if name in fit_guess:
                param_values[name] = fit_guess.get(name)
            else:
                raise ValueError(f"No initial value given for {name}")

            if name in fit_bounds:
                p.bounds = fit_bounds.get(name)
            param_bounds.append(p.bounds)

            if name in fit_constants:
                p.fixed = True
                fit_guess.pop(name)
            else:
                p.fixed = False
                param_names.append(name)

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
                        param_bounds,
                        list(param_values.values()),
                        opt_func
                )
            else:
                def condition(x, v=False):
                    params = param_info.get_namevaluepairs()
                    predict = circ_calc(params, 1e-13)
                    if v:
                        print(tot_imp - predict.real)
                    err = np.abs(tot_imp - predict.real)
                    return err

                def opt_func(x):
                    params = dict(zip(param_names, x))
                    param_values.update(params)
                    predict = circ_calc(param_values, frequencies)
                    last_predict = circ_calc(param_values, 1e-13)
                    err = 10 * rmse(predict, z) + np.abs(tot_imp - last_predict.real)
                    return err

                opt_result = fit_routine(
                        param_bounds,
                        list(param_values.values()),
                        opt_func
                )

            param_values = opt_result.x

        # print the fitting parameters to the console
        report = f"Fitting report:\n"
        report += f"Equivivalent circuit: {fit_circuit}\n"
        report += "Parameters: \n"
        for p_value, p_info in zip(param_values, param_info):
            p_info.value = p_value
            report += f"\t {p_info}\n"

        LOGGER.info(report)

        if path is not None:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, 'w') as f:
                json.dump(
                        param_info, f, default=lambda o: o.__dict__, indent=1
                )

        f_pred = np.logspace(-9, 9, 400)
        # plot the fitting result
        parameters = {info.name: info.value for info in param_info}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in tanh"
            )
            custom_circuit_fit_freq = circ_calc(parameters, frequencies)
            custom_circuit_fit = circ_calc(parameters, f_pred)

        ax.scatter(
                np.real(custom_circuit_fit_freq),
                -np.imag(custom_circuit_fit_freq),
                color="red",
                zorder=5,
                marker='x'
        )
        line = ax.plot(
                np.real(custom_circuit_fit),
                -np.imag(custom_circuit_fit),
                label="fit",
                color="red",
                zorder=5,
        )

        lines = {"fit": line}

        # check if circle needs to be drawn
        if draw_circle:
            self.plot_semis(fit_circuit, param_info, cell, ax)

        self.eis_params["fit_info"] = param_info
        return lines, param_info

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
