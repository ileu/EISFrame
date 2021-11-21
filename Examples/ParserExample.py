import numpy

from eisplottingtool.parser import parse_circuit

circuit_str = 'R_0-p(R-R1,C)-p(R,C)-Wo'

info, eqn = parse_circuit(circuit_str)
pars = {
    'R': 1,
    'R0': 2,
    'R1': 3,
    'C': 2,
    'W': 4,
    'omega': numpy.linspace(1, 6, 5, dtype=float)
    }
print(circuit_str)
print(eqn(pars))
