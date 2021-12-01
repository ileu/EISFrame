import re
from typing import Callable

import matplotlib.axes
import schemdraw as sd

from eisplottingtool.parser import circuit_components
from eisplottingtool.parser.CircuitComponents import Parameter
from schemdraw import dsp


def parse_circuit3(
        circ: str,
        draw: bool = False
        ) -> tuple[list[Parameter], Callable]:
    """ EBNF parser for a circuit string.

    Implements an extended Backus–Naur form to parse a string that descirbes
    a circuit.
    Already implemented circuit elements are locacted in CircuitComponents.py

    To put elements in series connect them through -.
    Parallel elements are created by p(Elm1, Elm2,...)

    The syntax of the EBNF is given by:

    circuit = element | element-circuit
    element = component | parallel
    parallel = p(circuit {,circuit})
    component = a circuit component

    Parameters
    ----------
    circ : str
        String that descirbes a circuit
    draw : bool
        If the circuit should be drawn. Default False.

    Returns
    -------
    param_info : dict
    calculate : Callable

    """
    param_info: list[Parameter] = []
    draw_list = []

    def component(c: str):
        """ process component and remove from circuit string c

        Parameters
        ----------
        c : str
            circuit string

        Returns
        -------

        """
        index = re.match(r'([a-zA-Z]+)_?\d?', c)
        name = c[:index.end()]
        c = c[index.end():]
        symbol = re.match('[A-Za-z]+', name).group()

        for key, comp in circuit_components.items():
            if comp.get_symbol() == symbol:
                break
        else:
            return c, 1

        param_info.extend(comp.get_parameters(name))
        return c, key + rf".calc(param,'{name}', omega)", comp

    def parallel(c: str):
        c = c[2:]
        tot_eq = ''
        comp_list = []
        while not c.startswith(')'):
            if c.startswith(','):
                c = c[1:]
            c, eq, comp = circuit(c)
            comp_list.append(comp)
            if tot_eq:
                tot_eq += " + "
            tot_eq += f"1.0 / ({eq})"
        c = c[1:]
        return c, f"1.0 / ({tot_eq})", comp_list

    def element(c: str):
        if c.startswith('p('):
            c, eq, comps = parallel(c)
        else:
            c, eq, comps = component(c)
        return c, eq, comps

    def circuit(c: str):
        if not c:
            return c, ''
        c, eq, comps = element(c)
        tot_eq = f"{eq}"
        comp_list = [comps]
        if c.startswith('-'):
            c, eq, comps = circuit(c[1:])
            tot_eq += f" + {eq}"
            comp_list.extend(comps)
        return c, tot_eq, comp_list

    __, equation, drawing = circuit(circ.replace(" ", ""))

    calculate = eval('lambda param, omega: ' + equation, circuit_components)
    return param_info, calculate, drawing


def drawer(drawing, d, s=1.0, direction=1):
    for i, draw in enumerate(drawing):
        if not isinstance(draw, list):
            d += draw.draw().right().scale(s)
            continue

        d.push()
        if direction > 0:
            d += dsp.Line().up().scale(s)
        else:
            d += dsp.Line().down().scale(s)
        drawer(draw, d, s, -direction)
        if direction > 0:
            d += dsp.Line().down().scale(s)
        else:
            d += dsp.Line().up().scale(s)
        d.pop()


def main():
    matplotlib.use('TkAgg')
    circuit = 'R-p(C,Ws-p(R,R))-R'
    circuit2 = 'R-p(CPE,CPE)-p(R-R,Ws-p(R,R-p(R,R)),R)-p(C,C,C,C)-R'
    info, calc, drawing = parse_circuit3(circuit, draw=True)

    d = sd.Drawing()

    drawer(drawing, d)

    d.draw()

    # d = sd.Drawing()
    # d.push()
    # d += elm.Resistor().color('green')
    # d += elm.Resistor().scale(0.1).color('red')
    # d += elm.Resistor().color('blue')
    # d.pop()
    # d.move(0, -1)
    # d1 = sd.Drawing()
    # d1 += elm.ElementDrawing(d)
    # d += elm.ElementDrawing(d1)
    # print(elm.Resistor().length(25).get_bbox())
    # test = d.draw(showframe=True, show=False)
    # ax: matplotlib.axes.Axes = test.ax
    # ax.locator_params(axis='x', nbins=20)
    # ax.set_xticks(np.arange(*np.round(ax.get_xlim()), 0.5))
    # ax.set_xticks(np.arange(*np.round(ax.get_xlim()), 0.5) + 0.25, minor=True)
    # ax.set_yticks(np.arange(*np.round(ax.get_ylim()), 0.25))
    # ax.grid(which='both')
    # plt.show()


if __name__ == "__main__":
    main()
