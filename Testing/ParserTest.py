import numpy as np

from Parser.CircuitParser import parse_circuit

circuit_str = 'R0-p(R-R1,C)'

test, names, eqn = parse_circuit(circuit_str)
pars = {'R': 1, 'R0': 2, 'R1': 3, 'C': 2}
w = np.linspace(0.1, 5, 4)
print(circuit_str)
print(eqn)
print(test(pars, w))
