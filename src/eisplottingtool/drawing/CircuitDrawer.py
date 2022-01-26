import re
from typing import Callable

import schemdraw as sd
from schemdraw import dsp

from eisplottingtool.parser.CircuitComponents import circuit_components
from eisplottingtool.utils.UtilClass import Parameter


def draw_circuit(
        circ: str,
        scale_h: float = 0.25,
        par_connector_length: float = 0.25,
        scaling: Callable[[float], float] = None,
        color_dict: dict[str, str] = None,
        ) -> sd.Drawing:
    """ Modified version of CircuitParser.parse_circuit to draw the circuit.

    Parameters
    ----------
    circ : str
        String that descirbes a circuit
    scale_h : float
    par_connector_length : float
    scaling : Callable[[float], float]
    color_dict : ParameterDict

    Returns
    -------
    drawing : sd.Drawing

    """

    if scaling is None:
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
        color = 'black'

        if color_dict is not None:
            color = color_dict.get(name, color)

        symbol = re.match('[A-Za-z]+', name).group()

        for key, comp in circuit_components.items():
            if comp.get_symbol() == symbol:
                break
        else:
            return c, 1

        nonlocal drawing
        drawing += comp.draw().right().color(color).scale(s)
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
