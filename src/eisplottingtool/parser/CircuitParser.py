import re
from typing import Callable
from eisplottingtool.parser.CircuitComponents import circuit_components


class Parameter:
    def __init__(self, name):
        self.name = name
        self.value = 0
        self.bounds = (0, 0)
        self.unit = ''

    def __str__(self):
        return

    def __repr__(self):
        return

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.name == other.name
        return False


def parse_circuit(circ: str) -> tuple[list, Callable]:
    """ EBNF parser for a circuit string.

    Implements an extended Backus–Naur form to parse a string that descirbes
    a circuit.
    Already implemented circuit elements are locacted in CircuitComponents.py

    To put elements in series connect them through -.
    Parallel elements are created by p(Elm1, Elm2,...)

    The syntax of the EBNF is given by:

    circuit = element | element-circuit
    element = component | parallel
    parallel = p(circuit {,circuit})
    component = a circuit component

    Parameters
    ----------
    circ : str
        String that descirbes a circuit

    Returns
    -------
    param_info : dict
    calculate : Callable

    """
    param_names = []
    param_units = []
    param_bounds = []

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

        for key, comp in circuit_components.items():
            symbol = re.match('[A-Za-z]+', name).group()
            if comp.get_symbol() == symbol:
                break
        else:
            return c, 1

        param_names.extend(comp.get_paramname(name))
        param_units.extend(comp.get_unit())
        param_bounds.extend(comp.get_bounds())
        return c, key + rf".calc(param,'{name}', omega)"

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

    calculate = eval('lambda param, omega: ' + equation, circuit_components)
    param_info = list(zip(param_names, param_bounds, param_units))
    return param_info, calculate
