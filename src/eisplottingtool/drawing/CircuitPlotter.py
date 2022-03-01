import warnings

import numpy as np

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
