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
from typing import TypeVar

import eclabfiles as ecf
import pandas as pd
import pint
from matplotlib import axes, pyplot as plt
from matplotlib.patches import BoxStyle
from matplotlib.ticker import AutoMinorLocator

from .utils import plot_legend
from .utils.UtilClass import Cell

LOGGER = logging.getLogger(__name__)
T = TypeVar('T', bound='Parent')

control_types = {
    0: "CC",
    3: "CR",
    4: "Rest",
    8: "PEIS"
}


def load_df_from_path(path):
    return None


def _get_default_data_param(columns):
    col_names = {}
    for col in columns:
        if match := re.match(r'Ewe[^|]*', col):
            col_names['voltage'] = match.group()
        elif match := re.match(r'I[^|]*', col):
            col_names['current'] = match.group()
        elif match := re.match(r'Re\(Z(we-ce)?\)[^|]*', col):
            col_names['real'] = match.group()
        elif match := re.match(r'-Im\(Z(we-ce)?\)[^|]*', col):
            col_names['imag'] = match.group()
        elif match := re.match(r'Phase\(Z(we-ce)?\)[^|]*', col):
            col_names['phase'] = match.group()
        elif match := re.match(r'\|Z(we-ce)?\|[^|]*', col):
            col_names['abs'] = match.group()
        elif match := re.match(r'time[^|]*', col):
            col_names['time'] = match.group()
        elif match := re.match(r'(z )?cycle( number)?[^|]*', col):
            col_names['cycle'] = match.group()
        elif match := re.match(r'freq[^|]*', col):
            col_names['frequency'] = match.group()
        elif match := re.match(r'Ns', col):
            col_names['Ns'] = match.group()
    return col_names


class EISFrame(pd.DataFrame):
    """
       EISFrame stores EIS data and plots/fits the data.
    """

    def __init__(
            self,
            name: str = None,
            path: str = None,
            df: pd.DataFrame = None,
            **kwargs
            ) -> None:
        """ Initialises an EISFrame

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
        super().__init__()
        self.eis_params = kwargs
        self.raw_df = df
        self.df = df

        if name is not None:
            self.eis_params["name"] = name

        if path is not None:
            self.eis_params["path"] = path

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()

    def __getitem__(self, item):
        self.eis_params['selection'] = item
        self.df = self.select_data(item)
        return self

    def select_data(self, selection):
        if isinstance(selection, int):
            return self.raw_df[selection]
        elif isinstance(selection, tuple):
            return self.raw_df[selection]
        elif isinstance(selection, dict):
            cyc = selection.get("cycle")
            ns = selection.get("sequence")
            if ns and cyc:
                return self.raw_df[(cyc, ns)]
            elif ns:
                return self.raw_df[(slice(None, ns))]
            elif cyc:
                return self.raw_df[cyc]
        else:
            raise ValueError("Invalid Selection")

    def load(self, path=None):
        # TODO: combine two or more files together
        if path is None:
            path = self.eis_params["path"]

        ext = os.path.splitext(path)[1][1:]

        if ext in {"csv", 'txt'}:
            data = pd.read_csv(path, sep=',', encoding='unicode_escape')
        elif ext in {'mpr', 'mpt'}:
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
        for p in data.attrs['params']:
            ctrl.append(p["ctrl_type"])

        self.df = data.sort_index()
        self.eis_params.update(col_names)


    def plot_nyquist(
            self,
            ax: axes.Axes = None,
            selection=None,
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
        if selection is None:
            selection = self.eis_params.get("selection")


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
