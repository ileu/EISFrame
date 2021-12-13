import re

import schemdraw as sd
import schemdraw.dsp as dsp

from eisplottingtool.parser import circuit_components


def parse_circuit3(
        circ: str,
        draw: bool = False
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

    drawing = sd.Drawing()

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
        nonlocal drawing
        drawing += comp.draw().right().color('blue')
        return c

    def measure_circuit(s: str):
        par = 0.0
        commas = 1.0
        length = []
        l = 1.0
        for c in s:
            if c == '-':
                l += 1.0
            elif c == ',' and par == 0.0:
                length.append(l)
                l = 1.0
                commas += 1.0
            elif c == 'p':
                par += 1.0
                l += 0.5
            elif c == ')':
                par -= 1
            if par < 0.0:
                break
        length.append(l)
        return commas, length

    def measure_circuit_2(s: str):
        level = 0.0
        height = 1.0
        level_length = {}
        current_length = 0.0
        for c in s:
            if c == '-':
                current_length += 1
            elif c == ',':
                if level_length.get(level, 0.0) <= current_length:
                    level_length[level] = current_length
            elif c == 'p':
                level_length[level] = current_length
                current_length = 0.5
                level += 1
            elif c == ')':
                if level_length.get(level, 0.0) <= current_length:
                    level_length[level] = current_length
                if level_length.get(level, 0.0) <= level_length.get(
                        level + 1,
                        0.0
                        ):
                    level_length[level] += level_length.get(level + 1, 0.0)
                level -= 1

            if level < 0:
                break
        return height, level_length[0.0]

    def parallel(c: str):
        nonlocal drawing
        c = c[2:]
        max_height, max_length = measure_circuit_2(c)
        drawing += dsp.Line().right().length(0.25 * drawing.unit).color("magenta")
        i = 0.0
        print(f"start, {max_length=}, {max_height=}")
        while not c.startswith(')'):
            drawing.push()

            if c.startswith(','):
                c = c[1:]
            height = -(max_height - 1) * scale_h + 2 * scale_h * i
            _, length = measure_circuit_2(c)
            i += 1.0
            print(f"{height=}, {length=}, {c=}")
            if height > 0:
                drawing += dsp.Line().up().length(height * drawing.unit)
            elif height < 0:
                drawing += dsp.Line().down().length(-height * drawing.unit)

            if length < max_length:
                drawing += dsp.Line().right().length(
                        0.5 * drawing.unit * (max_length - length)
                        )
            c = circuit(c)
            if length < max_length:
                drawing += dsp.Line().right().length(
                        0.5 * drawing.unit * (max_length - length)
                        )

            if height > 0:
                drawing += dsp.Line().down().length(height * drawing.unit)
            elif height < 0:
                drawing += dsp.Line().up().length(-height * drawing.unit)
            drawing.pop()
        drawing.move(max_length * drawing.unit, 0)
        drawing += dsp.Line().right().length(0.25 * drawing.unit).color('lime')
        c = c[1:]
        print("end")
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

    return drawing


def main():
    circuit1 = 'p(R,R,R,R)-p(R,R,R)'
    circuit2 = 'R-p(C,Ws-p(R,R))-p(R,R,R)'
    circuit3 = 'R-p(R-R,Ws-p(R,R-p(R,R)),R)-R'  # -p(C,C,C,C)-R'
    circuit4 = 'R-p(R-p(p(p(R,R),C-R-CPE),R),R,R,R,R,R,R,R,' \
              'CPE)-R-R-R-R-R-R-R-p(R,R,C)'
    circuits = [circuit4]

    for circ in circuits:
        print(10 * '-')
        print(f"{circ=}")
        drawing = parse_circuit3(circ)
        drawing.draw()
    # info, calc, d = parse_circuit3(circuit, draw=True)
    # d += elm.Resistor().right()
    # d.draw()


if __name__ == "__main__":
    main()
