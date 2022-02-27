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

logger = logging.getLogger(__name__)


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

        **kwargs
            circuit: str
                Equivalent circuit for impedance
            cell: Cell
                Battery cell for normalizing
        """
        self.eis_params = kwargs
        self._df = df
        self.mark_points = default_mark_points
        self.eis_params["Lines"] = {}

        if name is not None:
            self.eis_params["name"] = name

        if path is not None:
            self.eis_params["path"] = path

    def __str__(self):
        if self._df is None:
            self.load()
        return self._df.__str__()

    def __repr__(self):
        if self._df is None:
            self.load()
        return self._df.__repr__()

    def __getitem__(self, item):
        self.eis_params["selection"] = item
        if self._df is None:
            self.load()

        if isinstance(item, int):
            return EISFrame(df=self._df.loc[item], **self.eis_params)
        elif isinstance(item, tuple):
            return EISFrame(df=self._df.loc[item], **self.eis_params)
        elif isinstance(item, dict):
            cyc = item.get("cycle")
            ns = item.get("sequence")
            if ns and cyc:
                return EISFrame(df=self._df.loc[(cyc, ns)], **self.eis_params)
            elif ns:
                return EISFrame(df=self._df.loc[(slice(None, ns))], **self.eis_params)
            elif cyc:
                return EISFrame(df=self._df.loc[cyc], **self.eis_params)
        elif isinstance(item, str):
            if item in self._df:
                return self._df[item]
            elif item in self.eis_params:
                return self.eis_params[item]
        else:
            raise ValueError("Invalid Selection")

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df

    @property
    def time(self) -> np.array:
        if self._df is None:
            self.load()
        return self._df[self.eis_params["time"]].values

    @property
    def impedance(self) -> np.array:
        if self._df is None:
            self.load()
        value = self._df[self.eis_params["real"]].values
        value += -1j * self._df[self.eis_params["real"]].values
        return value

    @property
    def real(self) -> np.array:
        if self._df is None:
            self.load()
        return self._df[self.eis_params["real"]].values

    @property
    def imag(self) -> np.array:
        if self._df is None:
            self.load()
        return self._df[self.eis_params["imag"]].values

    @property
    def frequency(self) -> np.array:
        if self._df is None:
            self.load()
        return self._df[self.eis_params["frequency"]].values

    @property
    def current(self) -> np.array:
        if self._df is None:
            self.load()
        return self._df[self.eis_params["current"]].values

    @property
    def voltage(self) -> np.array:
        if self._df is None:
            self.load()
        return self._df[self.eis_params["voltage"]].values

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

        # ctrl = []
        # for p in data.attrs["params"]:
        #     ctrl.append(p["ctrl_type"])

        self._df = data.sort_index()
        self.eis_params.update(col_names)

    def modify_data(self):
        # TODO: maybe add transform as argument
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
        if self._df is None:
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
        if all(ax.get_legend_handles_labels()):
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

    def manipulatze(self, manipulate):
        df = manipulate(self._df.copy())
        return EISFrame(df, **self.eis_params)

    def fit_nyquist(
        self,
        circuit: str = None,
        initial_values: dict[str, float] = None,
        fit_bounds: dict[str, tuple] = None,
        fit_constants: list[str] = None,
        upper_freq: float = np.inf,
        lower_freq: float = 0,
        path: str = None,
        **kwargs,
    ) -> dict:
        """
        Fitting function for electrochemical impedance spectroscopy (EIS) data.

        For the fitting a model or equivilant circuit is needed. The equivilant circuit is defined as a string.
        To combine elements in series a dash (-) is used. Elements in parallel are wrapped by p( , ).
        An element is definied by an identifier (usually letters) followed by a digit.
        Already implemented elements are located in :class:`circuit_components<circuit_utils.circuit_components>`:

        +------------------------+--------+-----------+---------------+--------------+
        | Name                   | Symbol | Paramters | Bounds        | Units        |
        +------------------------+--------+-----------+---------------+--------------+
        | Resistor               | R      | R         | (1e-6, 1e6)   | Ohm          |
        +------------------------+--------+-----------+---------------+--------------+
        | Capacitance            | C      | C         | (1e-20, 1)    | Farrad       |
        +------------------------+--------+-----------+---------------+--------------+
        | Constant Phase Element | CPE    | CPE_Q     | (1e-20, 1)    | Ohm^-1 s^a   |
        |                        |        +-----------+---------------+--------------+
        |                        |        | CPE_a     | (0, 1)        |              |
        +------------------------+--------+-----------+---------------+--------------+
        | Warburg element        | W      | W         | (0, 1e10)     | Ohm^-1 s^0.5 |
        +------------------------+--------+-----------+---------------+--------------+
        | Warburg short element  | Ws     | Ws_R      | (0, 1e10)     | Ohm          |
        |                        |        +-----------+---------------+--------------+
        |                        |        | Ws_T      | (1e-10, 1e10) | s            |
        +------------------------+--------+-----------+---------------+--------------+
        | Warburg open elemnt    | Wo     | Wo_R      | (0, 1e10)     | Ohm          |
        |                        |        +-----------+---------------+--------------+
        |                        |        | Wo_T      | (1e-10, 1e10) | s            |
        +------------------------+--------+-----------+---------------+--------------+

        Additionaly an initial guess for the fitting parameters is needed.
        The initial guess is given as a dictionary where each key is the parameters name and
        the coresponding value is the guessed value for the circuit.

        The bounds of each paramater can be customized by the ``fit_bounds`` parameter.
        This parameter is a dictionary, where each key is the parameter name
         and the value constists of a tuple for the lower and upper bound (lb, ub).

        To hold a parameter constant, add the name of the paramter to a list and pass it as ``fit_constants``

        Parameters
        ----------
        df
            Dataframe with the impedance data

        real
            column label of the real part of the impedance

        imag
            column label of the imaginary part of the impedance

        freq
            column label of the frequency of the impedance

        circuit
            Equivalent circuit for the fit

        initial_values
            dictionary with initial values
            Structure: {"param name": value, ... }

        name
            the name of the fit

        fit_bounds
            Custom bounds for a parameter if default bounds are not wanted
            Structure: {"param name": (lower bound, upper bound), ...}
            Default is ''None''
        fit_constants
            list of parameters which should stay constant during fitting
            Structure: ["param name", ...]
            Default is ''None''

        ignore_neg_res
            ignores impedance values with a negative real part

        upper_freq:
            upper frequency bound to be considered for fitting

        lower_freq:
            lower frequency boudn to be considered for fitting
        repeat
            how many times ``fit_routine`` gets called
        """
        # load and prepare data
        frequencies = self.frequency

        mask = np.logical_and(lower_freq < frequencies, frequencies < upper_freq)
        mask = np.logical_and(mask, self.real > 0)

        frequency = frequencies[mask]

        z = self.real[mask] - 1j * self.imag[mask]
        # check and parse circuit

        if circuit:
            fit_circuit = circuit
        elif "circuit" in self.eis_params:
            fit_circuit = self.eis_params["circuit"]
        else:
            raise ValueError("No fit circuit given")

        param_info, circ_calc = parse_circuit(fit_circuit)

        if fit_bounds is None:
            fit_bounds = {}

        if fit_constants is None:
            fit_constants = []

        param_values = initial_values.copy()  # stores all the values of the parameters
        variable_names = []  # name of the parameters that are not fixed
        variable_guess = []  # guesses for the parameters that are not fixed
        variable_bounds = []  # bounds of the parameters that are not fixed

        for p in param_info:
            p_name = p.name
            if p_name in initial_values:
                if p_name not in fit_constants:
                    variable_bounds.append(fit_bounds.get(p_name, p.bounds))
                    variable_guess.append(initial_values.get(p_name))
                    variable_names.append(p_name)
            else:
                raise ValueError(f"No initial value given for {p_name}")

        # calculate the weight of each datapoint
        def weight(error, value):
            """calculates the absolute value squared and divides the error by it"""
            square_value = value.real ** 2 + value.imag ** 2
            return np.true_divide(error, square_value)

        # calculate rmse
        def rmse(predicted, actual):
            """Calculates the root mean squared error between two vectors"""
            e = np.abs(np.subtract(actual, predicted))
            se = np.square(e)
            wse = weight(se, actual)
            mse = np.nansum(wse)
            return np.sqrt(mse)

        # prepare optimizing function:
        def condition(params):
            #TODO
            res = params["R2"] + params["Wss1_R"]
            return 0 * res

        def opt_func(x: list[float]):
            params = dict(zip(variable_names, x))
            param_values.update(params)
            predict = circ_calc(param_values, frequency)
            main_err = rmse(predict, z)
            cond_err = condition(param_values)
            return main_err + cond_err

        # fit
        opt_result = fit_routine(
            opt_func,
            variable_guess,
            variable_bounds,
        )

        # update values in ParameterList
        param_values.update(dict(zip(variable_names, opt_result.x)))

        # # print the fitting parameters to the console
        # report = f"Fitting report:\n"
        # report += f"Equivivalent circuit: {fit_circuit}\n"
        # report += "Parameters: \n"
        # for p_value, p_info in zip(param_values, param_info):
        #     p_info.value = p_value
        #     report += f"\t {p_info}\n"
        #
        # LOGGER.info(report)

        if path is not None:
            logger.info(f"Wrote fit parameters to '{path}'")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(param_values, f, indent=1)

        self.eis_params["fit_info"] = param_info
        return param_values

    def plot_semis(
        self, circuit: str, param_info: list, cell=None, ax: axes.Axes = None
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

            if match := re.match(r"(?=.*(R_?\d?))(?=.*(C(?:PE)?_?\d?))", e):
                res = param_values.get(match.group(1))
                cap = [
                    param_values.get(key)
                    for key in param_values
                    if match.group(2) in key
                ]

                def calc_specific_freq(r, c, n=1):
                    return 1.0 / (r * c) ** n / 2 / np.pi

                specific_frequency = calc_specific_freq(res, *cap)

            elif match := re.match(r"(W[os]?_?\d?)", e):
                war = [
                    param_values.get(key)
                    for key in param_values
                    if match.group(1) in key
                ]
                if len(war) == 2:
                    specific_frequency = 1.0 / war[1]
                else:
                    specific_frequency = 1e-2
            elif re.match(r"(R_?\d?)", e):
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
                color = min(self.mark_points, key=lambda x: x.magnitude).color
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
                ls="None",
            )
            prev_imp += np.real(elem_impedance)[0]

        return


