import re

import numpy as np


class Component:
    def __init__(self, key):
        self.key = key

    def get_paramnames(self):
        return self.key

    def __call__(self):
        raise NotImplementedError


class Resistor(Component):
    def __call__(self):
        return self.key


class Capacitor(Component):
    def __call__(self):
        return f"(1j * omega * {self.key}) ** -1"


class CPE(Component):
    def __init__(self, key):
        super().__init__(key)  # not necessary but I guess good code??
        self.key = [self.key + "_0", self.key + "_1"]

    def __call__(self):
        return f"(1j * omega * {self.key[0]}) ** -{self.key[1]}"


def parse_circuit(circ):
    param_names = []

    def component(c: str):
        index = re.match(r'([A-z]+)_?\d?', c)
        key = c[:index.end()]
        c = c[index.end():]

        # TODO: depending on key select appropriate component
        if key.startswith('R'):
            comp = Resistor(key)
        elif key.startswith('C'):
            comp = Capacitor(key)
        else:
            return c, 1

        param_names.append(comp.get_paramnames())
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
            tot_eq += f"({eq}) ** -1"
        c = c[1:]
        return c, f"({tot_eq}) ** -1"

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
        return np.full(omega.shape, result)

    return evaluate, param_names, equation


circuit_str = 'R0-p(R-R1,C)'

test, names, eqn = parse_circuit(circuit_str)
pars = {'R': 1, 'R0': 2, 'R1': 3, 'C': 2}
w = np.linspace(0.1, 5, 4)
print(circuit_str)
print(eqn)
print(test(pars, w))
