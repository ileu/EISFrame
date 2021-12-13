import logging
import re

def measure_circuit_2(s: str, local=False):
    level = 0.0
    height = 1.0
    total_length = 0.0
    length = 0.0
    while level >= 0.0 and s != ')':
        s, c = s[1:], s[0]
        if c == '-':
            length += 1.0
        elif c == ',':
            if total_length < length:
                total_length = length
            length = 0.0
            height += 1.0
            if local and level == 0:
                break
        elif c == '(':
            length += 0.5
            __, par_length, s = measure_circuit_2(s)
            length += par_length
        elif c == ')':
            length += 1.0
            level -= 1
        elif match := re.match(r'^\w*', s):
            print(f"{c=}")
            print(match)

    if total_length < length:
        total_length = length
    return height, total_length, s


def main():
    test_string = 'R-p(p(p(R,R),C-R-CPE),R),R)'
    test_string2 = 'p(R,R)'
    exp_length = 5.0

    height, length, __ = measure_circuit_2(test_string2)
    __ , local_length, __ = measure_circuit_2(test_string2, True)

    print(f"{length=}, {exp_length=}, {local_length=}")
    print(f"{height=}")


if __name__ == '__main__':
    logging.info("start")
    main()
    logging.info("stop")
