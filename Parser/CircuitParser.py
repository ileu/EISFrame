import re

import numpy as np

from Parser.CircuitElements import circuit_components


def parse_circuit(circ):
    param_names = []
    param_units = []

    def component(c: str):
        index = re.match(r'([a-zA-Z]+)_?\d?', c)
        key = c[:index.end()]
        c = c[index.end():]

        for comp in circuit_components.values():
            symbol = re.match('[A-Za-z]+', key).group()
            if comp.get_symbol() == symbol:
                comp = comp(key)
                break
        else:
            return c, 1

        param_names.extend(comp.get_paramnames())
        param_units.append(comp.get_unit())
        return c, comp()

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

    __, equation = circuit(circ)

    return param_names, equation
