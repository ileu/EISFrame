import numpy

from Parser.CircuitParser import parse_circuit

circuit_str = 'R0-p(R-R1,C)-p(R,C)'

test, names, eqn = parse_circuit(circuit_str)
pars = {'R': 1, 'R0': 2, 'R1': 3, 'C': 2, 'W': 4}
w = numpy.linspace(1, 6, 5, dtype=float)
print(circuit_str)
print(eqn)
print(test(pars, w))
