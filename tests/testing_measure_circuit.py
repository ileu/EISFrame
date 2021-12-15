import logging
import re


def measure_circuit_2(s: str, local=False):
    height = 1.0
    total_length = 0.0
    length = 0.0
    while s != ')' and s != '':
        s, c = s[1:], s[0]
        if c == ',':
            if total_length < length:
                total_length = length
            length = 0.0
            height += 1.0
            if local:
                break
        elif c == '(':
            __, par_length, s = measure_circuit_2(s)
            length += par_length + 0.5
        elif not c.startswith("p") and c.isalpha():
            rest_of_element = re.match(r'^\w*', s)
            s = s[rest_of_element.end():]
            length += 1
        elif c == ')':
            break

    if total_length < length:
        total_length = length
    return height, total_length, s


def main():
    test_string = 'R-p(p(p(R,R),C-R-CPE),R),R,R)-R'
    test_string2 = 'p(R,R)'
    exp_length = 5.0

    height, length, __ = measure_circuit_2(test_string)
    __ , local_length, __ = measure_circuit_2(test_string, True)

    print(f"{length=}, {exp_length=}, {local_length=}")
    print(f"{height=}")


if __name__ == '__main__':
    logging.info("start")
    main()
    logging.info("stop")
