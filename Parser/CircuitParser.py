import re

import numpy as np

from CircuitElements import circuit_components


def parse_circuit(circ):
    param_names = []
    param_units = []

    def component(c: str):
        index = re.match(r'([A-z]+)_?\d?', c)
        key = c[:index.end()]
        c = c[index.end():]

        for comp in circuit_components.values():
            symbol = re.match('[A-z]+', key).group()
            if comp.get_symbol() == symbol:
                comp = comp(key)
                break
        else:
            return c, 1

        param_names.append(comp.get_paramnames())
        param_names.append(comp.get_unit())
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
            tot_eq += f"({eq}) ** -1.0"
        c = c[1:]
        return c, f"({tot_eq}) ** -1.0"

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

    def evaluate(params, omega):
        params["omega"] = omega
        result = eval(equation, params)
        result = np.array(result)
        omega = np.array(omega)
        if result.shape == omega.shape:
            return result
        elif not result.shape:
            return np.full(omega.shape, result)

        raise ValueError

    return evaluate, param_names, equation
