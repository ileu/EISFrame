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
    scale_h = 0.25

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
        return c

    def parallel_counter(s: str):
        par = 0
        ser = True
        commas = 1
        length = 1
        l = 1
        for c in s:
            if c == '-' and ser:
                l += 1
            elif c == ',':
                ser = False
                l = 1
                if par == 0:
                    commas += 1
            elif c == 'p':
                ser = False
                par += 1
            elif c == ')':
                ser = False
                par -= 1
            else:
                ser = True
            if l > length:
                length = l
            if par < 0:
                break
        return commas, length

    def parallel(c: str):
        c = c[2:]
        max_length, max_height = parallel_counter(c)
        i = 0
        print(f"s, {max_length=}, {max_height=}")
        while not c.startswith(')'):
            height = -(max_height - 1) * scale_h + 2 * scale_h * i
            i += 1
            print(height)
            if c.startswith(','):
                c = c[1:]
            c = circuit(c)
        c = c[1:]
        print("e")
        return c

    def element(c: str):
        if c.startswith('p('):
            c = parallel(c)
        else:
            c = component(c)
        return c

    def circuit(c: str):
        if not c:
            return c, ''
        c = element(c)
        if c.startswith('-'):
            c = circuit(c[1:])
        return c

    circuit(circ.replace(" ", ""))


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
    circuit = 'R-p(R-p(p(p(R,R),C-R-CPE),R),R,R,R,R,R,R,R,CPE)-R-R-R-R-R-R-R-p(R,R,C)'
    circuit1 = 'R-p(C,Ws-p(R,R))-R'
    circuit2 = 'R-p(CPE,CPE)-p(R-R,Ws-p(R,R-p(R,R)),R)-p(C,C,C,C)-R'
    circuits = [circuit1]

    for circ in circuits:
        print(10*'-')
        parse_circuit3(circ)
    # info, calc, d = parse_circuit3(circuit, draw=True)
    # d += elm.Resistor().right()
    # d.draw()


if __name__ == "__main__":
    main()
