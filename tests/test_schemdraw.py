import numpy as np
import pytest


def draw(drawer):
    length = len(drawer)
    result = []

    for l in np.arange(-length, length, 2) * 0.25 + 0.25:
        result.append(l)

    return result


@pytest.mark.parametrize(
        "elem, expected",
        [
            pytest.param([''], [0]),
            pytest.param(['', ''], [-0.25, 0.25]),
            pytest.param(['', '', ''], [-0.5, 0.0, 0.5]),
            pytest.param(['', '', '', ''], [-0.75, -0.25, 0.25, 0.75]),
            ]
        )
def test_height(elem, expected):
    actual = draw(elem)
    print(actual)
    assert actual == expected
