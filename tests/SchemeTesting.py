import re
from typing import Callable

import schemdraw as sd
import schemdraw.dsp as dsp
from eisplottingtool.parser import circuit_components
from eisplottingtool.parser.CircuitComponents import Parameter
from schemdraw import elements as elm


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
        d += comp.draw().length(d.unit * 0.75)
        return c, key + rf".calc(param,'{name}', omega)", d

    def parallel(c: str):
        c = c[2:]
        tot_eq = ''
        ed_list = []
        while not c.startswith(')'):
            if c.startswith(','):
                c = c[1:]
            c, eq, ed = circuit(c)
            ed_list.append(ed)
            if tot_eq:
                tot_eq += " + "
            tot_eq += f"1.0 / ({eq})"
        c = c[1:]
        d = sd.Drawing(fontsize=14)
        draw_parallel(ed_list, d)
        return c, f"1.0 / ({tot_eq})", d

    def element(c: str):
        if c.startswith('p('):
            c, eq, ed = parallel(c)
        else:
            c, eq, ed = component(c)
        return c, eq, ed

    def circuit(c: str):
        if not c:
            return c, ''
        c, eq, ed = element(c)
        tot_eq = f"{eq}"
        d = sd.Drawing(fontsize=14)
        d += elm.ElementDrawing(ed)
        if c.startswith('-'):
            c, eq, ed = circuit(c[1:])
            tot_eq += f" + {eq}"
            d += elm.ElementDrawing(ed)
        return c, tot_eq, d

    __, equation, drawing = circuit(circ.replace(" ", ""))

    calculate: Callable = eval(
            'lambda param, omega: ' + equation,
            circuit_components
            )
    return param_info, calculate, drawing


def draw_parallel(elements: list[sd.Drawing], d: sd.Drawing):
    d += dsp.Line().right().length(d.unit / 8.0)
    max_length = max(element.get_bbox().xmax for element in elements)
    if max_length % d.unit:
        max_length = max_length + (d.unit - max_length % d.unit)
    for i in range(len(elements)):
        height = -0.25 * (len(elements) - 1.0) + 0.5 * i
        d.push()
        if height >= 0:
            d += dsp.Line().up().length(d.unit * height)
        else:
            d += dsp.Line().down().length(d.unit * -height)
        if (length := elements[i].get_bbox().xmax) != max_length:
            length = (max_length - length) / 2.0
            d += dsp.Line().right().length(d.unit * length / 3.0)
            d += elm.ElementDrawing(elements[i]).right()
            d += dsp.Line().right().length(d.unit * length / 3.0)
        else:
            d += elm.ElementDrawing(elements[i]).right()
        if height < 0:
            d += dsp.Line().up().length(d.unit * -height)
        else:
            d += dsp.Line().down().length(d.unit * height)
        d.pop()

    d.move(d.unit * max_length / 3, 0)
    d += dsp.Line().right().length(d.unit / 8.0)

    return d


def main():
    # d1 = sd.Drawing()
    # d1 += elm.Resistor().length(d1.unit * 0.75).label('R1')
    # d2 = sd.Drawing()
    # d2 += elm.Capacitor().length(d1.unit * 0.75).label('R1')
    # d3 = sd.Drawing()
    # d3 += elm.CPE().length(d1.unit * 0.75).label('R1')
    # d3 = sd.Drawing()
    # d3 += elm.Resistor().length(d1.unit * 0.75).label('R1')
    # draw_parallel([d1, d1], d3)
    # d = sd.Drawing(fontsize=14)
    # d += elm.Resistor().right().length(d1.unit * 0.75).label('R1')
    # draw_parallel([d1, d2, d2], d)
    # d += elm.Capacitor().right()
    # draw_parallel([d1, d3], d)
    # d.draw()

    circuit = 'R-p(R-p(p(p(R,R),C-R-CPE),R),R,R,R,R,R,R,R,CPE)-R-R-R-R-R-R-R-p(R,R,C)'
    info, calc, d = parse_circuit3(circuit, draw=True)
    d += elm.Resistor().right()
    d.draw()


if __name__ == "__main__":
    main()
