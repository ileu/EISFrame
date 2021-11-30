import re
from typing import Callable
from eisplottingtool.parser.CircuitComponents import circuit_components, \
    Parameter

import schemdraw as sd
import schemdraw.dsp as dsp


def parse_circuit(
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
        return c, key + rf".calc(param,'{name}', omega)"

    def parallel(c: str):
        c = c[2:]
        tot_eq = ''
        while not c.startswith(')'):
            if c.startswith(','):
                c = c[1:]
            c, eq = circuit(c)
            if tot_eq:
                tot_eq += " + "
            tot_eq += f"1.0 / ({eq})"
        c = c[1:]
        return c, f"1.0 / ({tot_eq})"

    def element(c: str):
        if c.startswith('p('):
            c, eq = parallel(c)
        else:
            c, eq = component(c)
        return c, eq

    def circuit(c: str):
        if not c:
            return c, ''
        c, eq = element(c)
        tot_eq = f"{eq}"
        if c.startswith('-'):
            c, eq = circuit(c[1:])
            tot_eq += f" + {eq}"
        return c, tot_eq

    __, equation = circuit(circ.replace(" ", ""))

    calculate = eval('lambda param, omega: ' + equation, circuit_components)
    return param_info, calculate


def _draw_parallel(elements: list, drawing: sd.Drawing):
    drawing += dsp.Line().right().length(drawing.unit / 8.0)
    drawing.move(drawing.unit * 0.75, 0)
    for i in range(len(elements)):
        length = -0.25 * (len(elements) - 1.0) + 0.5 * i
        drawing.push()
        if length >= 0:
            drawing += dsp.Line().up().length(drawing.unit * length)
        else:
            drawing += dsp.Line().down().length(drawing.unit * -length)
        drawing += elements[i]().left().length(drawing.unit * 0.75)
        if length < 0:
            drawing += dsp.Line().up().length(drawing.unit * -length)
        else:
            drawing += dsp.Line().down().length(drawing.unit * length)
        drawing.pop()

    drawing += dsp.Line().right().length(drawing.unit / 8.0)

    return drawing
