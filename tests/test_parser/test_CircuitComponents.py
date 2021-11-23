import numpy as np
import pytest
from eisplottingtool.parser.CircuitComponents import Resistor, Capacitor, \
    CPE, Warburg, WarburgOpen, WarburgShort
from numpy.testing import assert_array_equal

param_values = {
    'R': 1,
    'C': 1,
    'CPE_Q': 1,
    'CPE_n': 1,
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
            (1j * 2 * np.pi * freq * param_values['CPE_Q']) ** -param_values[
                'CPE_n'],
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
    assert_array_equal(actual, expected)
