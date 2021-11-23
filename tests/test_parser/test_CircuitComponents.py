import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from eisplottingtool.parser.CircuitComponents import Resistor, Capacitor, \
    CPE, Warburg, WarburgOpen, WarburgShort

param_values = {
    'R': 1,
    'C': 1,
    'CPE_Q': 1,
    'CPE_n': 0.9,
    'W': 1,
    'Ws_R': 1,
    'Ws_T': 1,
    'Wo_R': 1,
    'Wo_T': 1,
    }

freq = 1

component_tests = [
    pytest.param(Resistor, 'R', np.full_like(freq, param_values['R'])),
    pytest.param(
            Capacitor,
            'C',
            1.0 / (1j * 2 * np.pi * freq * param_values['C']),
            id="Capacitance"
            ),
    pytest.param(
            CPE,
            'CPE',
            0.0299206180239438 - 0.188911347368689j,
            id="CPE"
            ),
    pytest.param(
            Warburg, 'W',
            param_values['W'] * (1 - 1j) / np.sqrt(2 * np.pi * freq),
            id="Warburg"
            ),
    pytest.param(
            WarburgShort,
            'Ws',
            param_values['Ws_R'] /
            np.sqrt(1j * param_values['Ws_T'] * 2 * np.pi * freq) *
            np.tanh(np.sqrt(1j * param_values['Ws_T'] * 2 * np.pi * freq)),
            id="WarburgShort"
            ),
    pytest.param(
            WarburgOpen,
            'Wo',
            param_values['Wo_R'] /
            np.sqrt(1j * param_values['Wo_T'] * 2 * np.pi * freq) /
            np.tanh(np.sqrt(1j * param_values['Wo_T'] * 2 * np.pi * freq)),
            id="WarburgOpen"
            ),
    ]


@pytest.mark.parametrize(
        "comp, symbol, expected",
        component_tests
        )
def test_calc(comp, symbol, expected):
    actual = comp.calc(param_values, symbol, freq)
    # print('\n', f"Calculated value: {actual}")
    assert_array_almost_equal(actual, expected)
