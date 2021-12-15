import re

import schemdraw as sd
import schemdraw.dsp as dsp

from eisplottingtool.parser import circuit_components
from matplotlib import pyplot as plt


def parse_circuit3(
        circ: str,
        ) -> sd.Drawing:
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
    drawing : sd.Drawing

    """
    scale_h = 0.25
    par_connector_length = 0.25

    def scaling(old_s):
        new_s = old_s * 0.025
        return 1

    drawing = sd.Drawing()

    def component(c: str, s: float):
        """ process component and remove from circuit string c

        Parameters
        ----------
        c : str
            circuit string
        s : float
            scale of element

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
        nonlocal drawing
        drawing += comp.draw().right().scale(s)
        return c

    def measure_circuit(c: str, s: float, local=False):
        height = 1
        total_length = 0
        length = 0
        while c != ')' and c != '':
            c, char = c[1:], c[0]
            if char == ',':
                length = 0
                height += 1
                if local:
                    break
            elif char == '(':
                __, par_length, c = measure_circuit(c, scaling(s))
                length += par_length + 2 * par_connector_length * s
            elif not char.startswith("p") and char.isalpha():
                rest_of_element = re.match(r'^\w*', c)
                c = c[rest_of_element.end():]
                length += s
            elif char == ')':
                break

            if total_length < length:
                total_length = length

        return height, total_length, c

    def parallel(c: str, s: float):
        nonlocal drawing
        c = c[2:]
        new_s = scaling(s)

        max_height, max_length, __ = measure_circuit(c, 1)
        drawing += dsp.Line().right().length(
                par_connector_length * drawing.unit * s
                )
        i = 0

        while not c.startswith(')'):
            if c.startswith(','):
                c = c[1:]

            height = -(max_height - 1) * scale_h + 2 * scale_h * i
            height = height * new_s
            __, length, __ = measure_circuit(c, 1, local=True)
            i += 1

            drawing.push()

            if height > 0:
                drawing += dsp.Line().up().length(height * drawing.unit)
            elif height < 0:
                drawing += dsp.Line().down().length(-height * drawing.unit)

            if length < max_length:
                drawing += dsp.Line().right().length(
                        0.5 * drawing.unit * (max_length - length) * new_s
                        )

            c = circuit(c, new_s)

            if length < max_length:
                drawing += dsp.Line().right().length(
                        0.5 * drawing.unit * (max_length - length) * new_s
                        )
            if height > 0:
                drawing += dsp.Line().down().length(height * drawing.unit)
            elif height < 0:
                drawing += dsp.Line().up().length(-height * drawing.unit)
            drawing.pop()
        drawing.move(max_length * drawing.unit * new_s, 0)
        drawing += dsp.Line().right().length(
                par_connector_length * drawing.unit * s
                )
        c = c[1:]
        return c

    def element(c: str, s: float):
        if c.startswith('p('):
            c = parallel(c, s)
        else:
            c = component(c, s)
        return c

    def circuit(c: str, s: float):
        if not c:
            return c, ''
        c = element(c, s)
        if c.startswith('-'):
            c = circuit(c[1:], s)
        return c

    circuit(circ.replace(" ", ""), 1)

    return drawing


def main():
    circuit1 = 'p(R,R,R,R)-p(R,R,R)'
    circuit2 = 'R-p(C,Ws-p(R,R))-p(R,R,R)'
    circuit3 = 'R-p(R-R,Ws-p(R,R-p(R,R)),R)-R-p(C,C,C,C)-R'
    circuit4 = 'R-p(R-p(p(p(R,R),C-R-CPE),R),R,R,R,R,R,R,R,' \
               'CPE)-R-R-R-R-R-R-R-p(R,R,C)'
    circuit5 = 'p(p(R,R),R)'
    circuits = [circuit1, circuit2, circuit3, circuit4, circuit5]

    for circ in circuits:
        print(25 * '-')
        print(f"{circ=}")
        drawing = parse_circuit3(circ)
        drawing.draw(show=False)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()


if __name__ == "__main__":
    main()
