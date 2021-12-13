import logging


def measure_circuit_2(s: str):
    level = 0.0
    height = 1.0
    level_length = {}
    current_length = 0.0
    for c in s:
        if c == '-':
            current_length += 1
        elif c == ',':
            if level_length.get(level, 0.0) <= current_length:
                level_length[level] = current_length
        elif c == 'p':
            level_length[level] = current_length
            current_length = 0.5
            level += 1
        elif c == ')':
            if level_length.get(level, 0.0) <= current_length:
                level_length[level] = current_length
            if level_length.get(level, 0.0) <= level_length.get(level + 1, 0.0):
                level_length[level] += level_length.get(level + 1, 0.0)
            level -= 1

        if level < 0:
            break

    print(f"{level_length=}")
    return height, level_length[0.0]


def main():
    test_string = 'R-p(p(p(R,R),C-R-CPE),R),R)'
    exp_length = 5.0

    height, length = measure_circuit_2(test_string)

    print(f"{length=}, {exp_length=}")


if __name__ == '__main__':
    logging.info("start")
    main()
    logging.info("stop")
