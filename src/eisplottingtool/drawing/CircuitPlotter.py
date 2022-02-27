import warnings

import numpy as np

from ..parser import parse_circuit


def plot_fit(
    ax,
    circuit,
    param_info,
    frequencies=np.logspace(-9, 9, 400),
    kind: str = "line",
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
        custom_circuit_fit = circ_calc(parameters, frequencies)

    if kind == "scatter":
        plot = ax.scatter(
            np.real(custom_circuit_fit),
            -np.imag(custom_circuit_fit),
            color=kwargs.get("color"),
            zorder=5,
            marker="x",
        )
    elif kind == "line":
        plot = ax.plot(
            np.real(custom_circuit_fit),
            -np.imag(custom_circuit_fit),
            label="fit",
            color=kwargs.get("color"),
            zorder=5,
        )
    else:
        raise ValueError(f"Unknown kind for plot found: '{kind}'")
    lines = {"fit": plot}

    return lines