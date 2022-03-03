import re
import warnings

import numpy as np
from matplotlib import axes, pyplot as plt

from ..parser import parse_circuit
from ..utils import plot_legend


def plot_circuit(
    ax,
    circuit,
    param_info,
    frequencies=np.logspace(-9, 9, 400),
    manipulate=None,
    kind: str = "line",
    show_legend=True,
    **kwargs,
):
    __, circ_calc = parse_circuit(circuit)

    # plot the fitting result
    if isinstance(param_info, list):
        parameters = {info.name: info.value for info in param_info}
    else:
        parameters = param_info

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in tanh")
        data = circ_calc(parameters, frequencies)

    if manipulate:
        data = manipulate(data)

    if kind == "scatter":
        plot = ax.scatter(np.real(data), -np.imag(data), **kwargs)
    elif kind == "line":
        plot = ax.plot(np.real(data), -np.imag(data), **kwargs)
    else:
        raise ValueError(f"Unknown kind for plot found: '{kind}'")

    if show_legend:
        plot_legend(ax)

    lines = {"fit": plot}

    return lines

def plot_semis(
    circuit: str,
    param_values: dict,
    cell=None,
    ax: axes.Axes = None,
    manipulate=None,
    mark_points=None
):
    """
    plots the semicircles to the corresponding circuit elements.

    The recognised elements are p(R,CPE), p(R,C) or any Warburg element

    Parameters
    ----------
    mark_points
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
    # TODO No mark point case
    # check if axes is given, else get current axes
    if mark_points is None:
        mark_points = []
    if ax is None:
        ax = plt.gca()
    elem_infos = []

    if len(mark_points) == 0:
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
            color = min(mark_points, key=lambda x: x.magnitude).color
        else:
            for mark in mark_points:
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
