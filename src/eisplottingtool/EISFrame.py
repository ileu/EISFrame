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
from typing import Union

import numpy as np
import pandas as pd
import pint
from matplotlib import axes, pyplot as plt
from matplotlib.patches import BoxStyle
from matplotlib.ticker import AutoMinorLocator

from .parser import parse_circuit
from .utils import MarkPoint, plot_legend
from .utils.UtilFunctions import load_df_from_path
from .utils.fitting import fit_routine

logger = logging.getLogger(__name__)


class EISFrame:
    """
    EISFrame stores EIS data and plots/fits the data.
    """

    def __init__(
        self,
        name: str = None,
        path: Union[str, list[str]] = None,
        df: pd.DataFrame = None,
        **kwargs,
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
    def mark_points(self) -> list[MarkPoint]:
        return self.eis_params.get("mark_points") or []

    @mark_points.setter
    def mark_points(self, mp):
        self.eis_params["mark_points"] = mp

    def load(self, path=None):
        if path is None:
            path = self.eis_params["path"]

        if not isinstance(path, list):
            path = [path]

        data = []
        last_time = 0
        last_cycle = 0
        cycle = None
        for p in path:
            d = load_df_from_path(p)
            for col in d.columns:
                if match := re.match(r"(z )?cycle( number)?[^|]*", col):
                    cycle = match.group()
            if not cycle:
                raise ValueError(f"No cycles detected in file {p}.")
            d["cycle"] = d[cycle] + last_cycle
            d["time"] = d["time"] + last_time
            last_time = d["time"].iloc[-1]
            last_cycle = d["cycle"].iloc[-1]
            data.append(d)
        data = pd.concat(data)
        if "Ns changes" in data:
            data["technique"] = data.groupby(cycle)["Ns changes"].transform(
                pd.Series.cumsum
            )
            data.set_index(["cycle", "technique"], inplace=True)
        else:
            data.set_index(["cycle"], inplace=True)

        if "Re(Z)" not in data:
            data["Re(Z)"] = data.get("|Z|", 0) * np.cos(data.get("Phase(Z)", 0) / 360.0 * 2 * np.pi)
            data["-Im(Z)"] = -data.get("|Z|", 0) * np.sin(data.get("Phase(Z)", 0) / 360.0 * 2 * np.pi)

        self._df = data.copy()

    def manipulate(self, col_name, new_name, modify):
        if self._df is None:
            self.load()
        self._df[new_name] = modify(self._df[col_name].copy())
        return self

    def plot(self, nbinsx=6, nbinsy=4, title_xoffset=0.5, title_yoffset=0.9, title_hor_align="center", *args, **kwargs):
        if self._df is None:
            self.load()
        legend = kwargs.pop("legend", True)
        title = kwargs.pop("title", None)
        ax = kwargs.get("ax", plt.gca())
        plot = self._df.plot(legend=False, *args, **kwargs)

        ax.locator_params(axis="x", nbins=nbinsx)
        ax.locator_params(axis="y", nbins=nbinsy, prune="both")

        if title:
            ax.text(
                title_xoffset, title_yoffset, title, horizontalalignment=title_hor_align, transform=ax.transAxes
            )
        if legend:
            plot_legend(ax)
        return plot

    def plot_nyquist(
        self,
        ax: axes.Axes = None,
        exclude_data=None,
        show_freq: bool = False,
        color=None,
        ls="None",
        marker=None,
        plot_range=None,
        show_legend=True,
        show_mark_label=True,
        real_col=None,
        imag_col=None,
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
        exclude_data
        show_freq
        color
        ls
        marker
        plot_range
        show_legend
        show_mark_label
        real_col
        imag_col

        kwargs:
            color?

        Returns
        -------
        dictionary
            Contains all the matplotlib.lines.Line2D of the drawn plots
        """
        if self._df is None:
            self.load()

        frequency = self._df["freq"].to_numpy()

        if real_col:
            real = self._df[real_col].to_numpy()
        else:
            real = self._df["Re(Z)"].to_numpy()

        if imag_col:
            imag = self._df[imag_col].to_numpy()
        else:
            imag = self._df["-Im(Z)"].to_numpy()

        # only look at measurements with frequency data
        mask = frequency != 0

        if exclude_data is None:
            exclude_data = slice(None)
        # get the x,y data for plotting
        x_data = real[mask][exclude_data]
        y_data = imag[mask][exclude_data]
        frequency = frequency[mask][exclude_data]

        if "x_label" in kwargs:
            x_label = kwargs.get("x_label")
        # label for the plot
        else:
            x_label = rf"Re(Z) [Ohm]"

        if "y_label" in kwargs:
            y_label = kwargs.get("y_label")
        else:
            y_label = rf"-Im(Z) [Ohm]"

        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.eis_params.get("name")

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
                if mark.name:
                    mark_label = f"{mark.name} @ {mark.label(frequency)}"
                else:
                    mark_label = f"{mark.label(frequency)}"
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

    def fit_nyquist(
        self,
        circuit: str = None,
        initial_values: dict[str, float] = None,
        fit_bounds: dict[str, tuple] = None,
        fit_constants: list[str] = None,
        upper_freq: float = np.inf,
        lower_freq: float = 0,
        path: str = None,
        condition = None,
        **kwargs,
    ) -> dict:
        """
        Fitting function for electrochemical impedance spectroscopy (EIS) data.

        For the fitting a model or equivalent circuit is needed. The equivilant circuit is defined as a string.
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
        frequencies = self._df["freq"].to_numpy()
        real = self._df["Re(Z)"].to_numpy()
        imag = self._df["-Im(Z)"].to_numpy()

        mask = np.logical_and(lower_freq < frequencies, frequencies < upper_freq)
        mask = np.logical_and(mask, real > 0)

        frequency = frequencies[mask]

        z = real[mask] - 1j * imag[mask]
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
        if condition is None:
            def condition(params):
                return 0

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
        self,
        circuit: str,
        param_values: list,
        cell=None,
        ax: axes.Axes = None,
        manipulate=None,
        ignore_ecr=False,
    ):
        """
        plots the semicircles to the corresponding circuit elements.

        The recognised elements are p(R,CPE), p(R,C) or any Warburg element

        Parameters
        ----------
        circuit : CustomCircuit
            CustomCircuit
        param_values
        cell
        ax : matplotlib.axes.Axes
             axes to be plotted to

        Returns
        -------
        nothing at the moment
        """
        # TODO No mark point case, Ecr rework
        # check if axes is given, else get current axes
        if ax is None:
            ax = plt.gca()
        elem_infos = []

        if len(self.mark_points) == 0:
            warnings.warn("No Markpoints chosen, will draw circles with gray color")

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

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="overflow encountered in tanh"
                )
                elem_impedance = elem_eval(param_values, freq)

            if manipulate:
                elem_impedance = manipulate(elem_impedance)

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
                if ignore_ecr:
                    continue
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
