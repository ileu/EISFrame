import re

import numpy as np

from Parser.CircuitElements import circuit_components


def parse_circuit(circ):
    param_names = []
    param_units = []

    def component(c: str):
        index = re.match(r'([a-zA-Z]+)_?\d?', c)
        name = c[:index.end()]
        c = c[index.end():]

        for key, comp in circuit_components.items():
            symbol = re.match('[A-Za-z]+', name).group()
            if comp.get_symbol() == symbol:
                break
        else:
            return c, 1

        param_names.extend(comp.get_paramname(name))
        param_units.extend(comp.get_unit())
        return c, key + rf".calc(param,'{name}')"

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

    calculate = eval('lambda param: ' + equation, circuit_components)
    param_info = dict(zip(param_names, param_units))
    return param_info, calculate
