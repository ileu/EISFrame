import re

import numpy as np

from Parser.CircuitElements import circuit_components


def calc_circuit(param, circ, omega):

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

        return c, comp.calc(param)

    def parallel(c: str):
        c = c[2:]
        tot_eq = 0
        while not c.startswith(')'):
            if c.startswith(','):
                c = c[1:]
            c, eq = circuit(c)
            tot_eq += 1.0 / eq
        c = c[1:]
        return c, 1.0 / tot_eq

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
        tot_eq = eq
        if c.startswith('-'):
            c, eq = circuit(c[1:])
            tot_eq += eq
        return c, tot_eq

    param['omega'] = omega
    __, equation = circuit(circ)

    return equation
