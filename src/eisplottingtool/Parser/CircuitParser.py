import re
from typing import Callable
import numpy as np
from eisplottingtool.Parser.CircuitComponents import circuit_components


def parse_circuit(circ: str) -> tuple[dict, Callable[[dict], np.array]]:
    """ EBNF parser for a circuit string.

    Implements an extended Backus–Naur form to parse a string that descirbes
    a circuit.
    Already implemented circuit elements are locacted in CircuitComponents.py

    To put elements in series connect them through -.
    Parallel elements are created by p(Elm1, Elm2,...)

    The syntax of the EBNF is given by:

    circuit = element | element-circuit
    element = component | parallel
    parallel = p(circuit, {circuit})
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