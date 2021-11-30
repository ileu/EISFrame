import re
from typing import Callable

import matplotlib.axes
import numpy as np
import schemdraw as sd
import schemdraw.dsp as dsp
from matplotlib import pyplot as plt
from schemdraw import elements as elm

from eisplottingtool.parser import circuit_components
from eisplottingtool.parser.CircuitComponents import Parameter


def parse_circuit3(
        circ: str,
        draw: bool = False
        ) -> tuple[list[Parameter], Callable, sd.Drawing]:
    """ EBNF parser for a circuit string.

    Implements an extended Backus–Naur form to parse a string that descirbes
    a circuit.
    Already implemented circuit elements are locacted in CircuitComponents.py

    To put elements in series connect them through -.
    Parallel elements are created by p(Elm1, Elm2,...)

    The syntax of the EBNF is given by:

        - circuit = element | element-circuit
        - element = component | parallel
        - parallel = p(circuit {,circuit})
        - component = a circuit component

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
    drawing : sd.Drawing

    """
    param_info: list[Parameter] = []

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

        d = sd.Drawing(fontsize=14)
        d += comp.draw().length(2)
        return c, key + rf".calc(param,'{name}', omega)", d

    def parallel(c: str, s):
        c = c[2:]
        tot_eq = ''
        ed_list = []
        while not c.startswith(')'):
            if c.startswith(','):
                c = c[1:]
            c, eq, ed = circuit(c, s * 0.5)
            ed_list.append(ed)
            if tot_eq:
                tot_eq += " + "
            tot_eq += f"1.0 / ({eq})"
        c = c[1:]
        d = sd.Drawing(fontsize=14)
        d = draw_parallel(ed_list, d, s)
        return c, f"1.0 / ({tot_eq})", d

    def element(c: str, s):
        if c.startswith('p('):
            c, eq, ed = parallel(c, s)
        else:
            c, eq, ed = component(c)
        return c, eq, ed

    def circuit(c: str, s=1.0):
        if not c:
            return c, ''
        c, eq, ed = element(c, s)
        tot_eq = f"{eq}"
        d = sd.Drawing(fontsize=14)
        d += elm.ElementDrawing(ed)
        if c.startswith('-'):
            c, eq, ed = circuit(c[1:], s)
            tot_eq += f" + {eq}"
            d += elm.ElementDrawing(ed)
        return c, tot_eq, d

    __, equation, drawing = circuit(circ.replace(" ", ""))

    calculate: Callable = eval(
            'lambda param, omega: ' + equation,
            circuit_components
            )
    return param_info, calculate, drawing


def draw_parallel(elements: list[sd.Drawing], d: sd.Drawing, s):
    d += dsp.Line().right().length(0.25).color('purple')
    max_length = max(element.get_bbox().xmax for element in elements)
    print(10*"-")
    print(s)
    for i, element in enumerate(elements):
        height = - 0.5 * (len(elements) - 1) + i
        scale = 0.5 / (element.get_bbox().ymax - element.get_bbox().ymin)
        d.push()
        print(element.get_bbox())

        if height >= 0:
            d += dsp.Line().up().length(height)
        else:
            d += dsp.Line().down().length(-height)

        if (length := element.get_bbox().xmax) != max_length:
            length = (max_length - length) / 2.0 / scale
            print(f"{length=}")
            d += dsp.Line().right().length(length).color('blue')
            d += elm.ElementDrawing(element).right().scale(s).color('blue')
            d += dsp.Line().right().length(length).color('red')
            print("long")
        else:
            d += elm.ElementDrawing(element).right().scale(s).color('green')
            print("short")

        if height < 0:
            d += dsp.Line().up().length(-height).color('blue')
        else:
            d += dsp.Line().down().length(height)

        d.pop()

    d.move(max_length, 0)
    d += dsp.Line().right().length(0.25).color('yellow')
    print(f"{d.get_bbox()=}")
    return d


def main():
    matplotlib.use('TkAgg')
    circuit = 'R-p(R,Ws-p(R,R))-R'
    circuit2 = 'R-p(CPE,CPE)-p(R-R,Ws-p(R,R-p(R,R)),R)-p(C,C,C,C)-R'  # 'R-p(R,R),p(R,CPE))'
    info, calc, d = parse_circuit3(circuit, draw=True)

    d = sd.Drawing()
    d.push()
    d += elm.Resistor().color('green')
    d += elm.Resistor().scale(0.1).color('red')
    d += elm.Resistor().color('blue')
    d.pop()
    d.move(0, -1)
    d1 = sd.Drawing()
    d1 += elm.ElementDrawing(d)
    d += elm.ElementDrawing(d1)
    print(elm.Resistor().length(25).get_bbox())
    test = d.draw(showframe=True, show=False)
    ax: matplotlib.axes.Axes = test.ax
    ax.locator_params(axis='x', nbins=20)
    ax.set_xticks(np.arange(*np.round(ax.get_xlim()), 0.5))
    ax.set_xticks(np.arange(*np.round(ax.get_xlim()), 0.5) + 0.25, minor=True)
    ax.set_yticks(np.arange(*np.round(ax.get_ylim()), 0.25))
    ax.grid(which='both')
    plt.show()


if __name__ == "__main__":
    main()
