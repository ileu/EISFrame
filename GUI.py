import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import Base
from Parser.CircuitParser import parse_circuit


def main():
    ir0 = 28
    pixel = 1.0
    axcolor = 'lavender'
    x_label = r"Re(Z)/$\Omega$"
    y_label = r"-Im(Z)/$\Omega$"

    fig, ax = Base.create_fig(figsize=(16, 12))
    ax.set_position([0.05, 0.11, 0.52, 0.85])

    params = ["Test1", "Test2"]
    slider_axs = []
    sliders = []
    for i, param in enumerate(params):
        slid_ax = fig.add_axes(
                [0.65, 0.85 - i * 0.05, 0.3, 0.03], facecolor=axcolor
                )
        slider = Slider(slid_ax, param, 0, 50.0e-2)

        slider_axs.append(slid_ax)
        sliders.append(slider)

    circuit = r"R0-p(R1,C1)"
    evaluate, names, eqn = parse_circuit(circuit)
    pars = {'R0': 2e-2, 'R1': 6.79e-3, 'C1': 5.62, 'R2': 3.91 * 1e-3, 'C2': 5.88 * 1e-2}
    w = np.logspace(-1, 6, 200, dtype=float)
    data = evaluate(pars, w)
    data = np.array(data)

    line, = ax.plot(data.real, -data.imag)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xlim(-max(data.real) * 0.05, max(data.real) * 1.05)
    ax.set_ylim(*ax.get_xlim())
    ax.locator_params(nbins=4)
    ax.set_aspect('equal')

    def update(val=None):
        test = sliders[0].val

        pars = {
            'R0': test,
            'R1': 6.79e-3,
            'C1': 5.62,
            'R2': 3.91 * 1e-3,
            'C2': 5.88 * 1e-2
            }
        w = np.logspace(-1, 6, 200, dtype=float)
        data = evaluate(pars, w)
        data = np.array(data)

        line.set_xdata(data.real)
        line.set_ydata(-data.imag)

        # ax.set_xlabel(x_label)
        # ax.set_ylabel(y_label)
        #
        # ax.set_xlim(-max(data.real) * 0.05, max(data.real) * 1.05)
        # ax.set_ylim(*ax.get_xlim())
        # ax.locator_params(nbins=4)
        # ax.set_aspect('equal')

        fig.canvas.draw()

    sliders[0].on_changed(update)

    plt.show()


if __name__ == '__main__':
    print("start")
    main()
    print("end")
