import logging
import re

import pytest

par_connector_length = 0.25


def scaling(old_s):
    return 1


test_circuits = [
    pytest.param("p(R,R)", 1, 1.5),
    pytest.param("R,R)", 2, 1.0),
    pytest.param("R,R,R,R)", 4, 1),
    pytest.param("p(R,R,R,R)", 1, 1.5),
    pytest.param("p(R-p(p(p(R,R),R-R-R),R),R,R)-R", 1, 6.5),
    pytest.param("R-p(p(p(R,R),R-R-R),R),R,R)-R", 3, 5.0),
]


@pytest.mark.parametrize("circuit, exp_height, exp_length", test_circuits)
def test_circuit_measure(circuit, exp_height, exp_length):
    actual_height, actual_length, __ = measure_circuit(circuit)
    assert exp_height == actual_height
    assert exp_length == actual_length


def measure_circuit(c: str, s: float = 1.0, local=False):
    height = 1
    total_length = 0
    length = 0
    while c != ")" and c != "":
        c, char = c[1:], c[0]
        if char == ",":
            length = 0
            height += 1
            if local:
                break
        elif char == "(":
            __, par_length, c = measure_circuit(c, scaling(s))
            length += par_length + 2 * par_connector_length * s
        elif not char.startswith("p") and char.isalpha():
            rest_of_element = re.match(r"^\w*", c)
            c = c[rest_of_element.end() :]
            length += s
        elif char == ")":
            break

        if total_length < length:
            total_length = length

    return height, total_length, c


def main():
    test_string = "R-p(p(p(R,R),C-R-CPE),R),R,R)-R"
    test_string2 = "p(R,R)"
    exp_length = 5.0

    height, length, __ = measure_circuit(test_string)
    __, local_length, __ = measure_circuit(test_string, True)

    print(f"{length=}, {exp_length=}, {local_length=}")
    print(f"{height=}")


if __name__ == "__main__":
    logging.info("start")
    main()
    logging.info("stop")
