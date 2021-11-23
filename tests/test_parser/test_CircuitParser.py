import pytest
from eisplottingtool.parser import parse_circuit
from numpy.testing import assert_array_almost_equal

param_values = {
    'R': 1,
    'C': 1,
    'CPE_Q': 1,
    'CPE_n': 0.5,
    'W': 1,
    'Ws_R': 1,
    'Ws_T': 1,
    'Wo_R': 1,
    'Wo_T': 1,
    }

test_circuits = [
    pytest.param('R-R-R', 3, id="Series"),
    pytest.param('R-p(R,R)', 1.5, id="Parallel"),
    pytest.param('R-p(R,R)-R', 2.5, id="Mix1"),
    pytest.param('R-p(R-R,R)-R', 2 + 2.0 / 3.0, id="Mix2"),
    pytest.param('R-p(R,C)', 1.0247045 - 0.1552231j, id="Randles"),
    pytest.param('R-p(R,CPE)', 1.2560427 - 0.1636903j, id="RandlesCPE"),
    ]


@pytest.mark.parametrize(
        "circuit, expected",
        test_circuits
        )
def test_parse_circuit(circuit, expected):
    circuit_info, circuit_calc = parse_circuit(circuit)
    assert_array_almost_equal(
            circuit_calc(param_values, 1),
            expected
            )
