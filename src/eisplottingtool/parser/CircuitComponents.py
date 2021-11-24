import numpy as np

initial_state = set(globals().copy())
non_element_functions = ['Component', 'initial_state', 'non_element_functions',
                         'Circuit']


# TODO: add default bounds


class Circuit:
    def __init__(self, circ_string):
        self.circ_string = circ_string


class Component:
    @staticmethod
    def get_symbol():
        raise NotImplementedError

    @staticmethod
    def get_unit():
        raise NotImplementedError

    @staticmethod
    def get_paramname(name):
        return [name]

    @staticmethod
    def calc(param, key, freq):
        raise NotImplementedError

    @staticmethod
    def get_bounds():
        raise NotImplementedError


class Resistor(Component):
    @staticmethod
    def get_symbol():
        return 'R'

    @staticmethod
    def get_unit():
        return ['Ohm']

    @staticmethod
    def calc(param, key, freq):
        result = np.full_like(freq, param.get(key), dtype=float)
        return result

    @staticmethod
    def get_bounds():
        return [(0.01, 2000)]


class Capacitor(Component):
    @staticmethod
    def get_symbol():
        return 'C'

    @staticmethod
    def get_unit():
        return ['F']

    @staticmethod
    def calc(param, key, freq):
        value = param.get(key)
        return 1.0 / (1j * 2 * np.pi * freq * value)

    @staticmethod
    def get_bounds():
        return [(1e-15, 1)]


class CPE(Component):
    @staticmethod
    def get_symbol():
        return 'CPE'

    @staticmethod
    def get_unit():
        return 'Ohm^-1 sec^n', ''

    @staticmethod
    def get_paramname(name):
        return name + "_Q", name + "_n"

    @classmethod
    def calc(cls, param, key, freq):
        values = [param[name] for name in cls.get_paramname(key)]
        result = (1j * 2 * np.pi * freq) ** (-values[1]) / values[0]
        print("CPE", result)
        return result

    @staticmethod
    def get_bounds():
        return (1e-15, 1), (0, 1)


class Warburg(Component):
    """ defines a semi-infinite Warburg element    """
    @staticmethod
    def get_symbol():
        return 'W'

    @staticmethod
    def get_unit():
        return ['Ohm s^-1/2']

    @staticmethod
    def get_paramname(name):
        return name + "_R", name + "_R"

    @staticmethod
    def calc(param, key, freq):
        value = param.get(key)
        return value * (1 - 1j) / np.sqrt(2 * np.pi * freq)

    @staticmethod
    def get_bounds():
        return [(0, 2000)]


class WarburgOpen(Component):
    """ defines a semi-infinite Warburg element    """
    @staticmethod
    def get_symbol():
        return 'Wo'

    @staticmethod
    def get_unit():
        return 'Ohm', 's'

    @staticmethod
    def get_paramname(name):
        return name + "_R", name + "_T"

    @classmethod
    def calc(cls, param, key, freq):
        values = [param[name] for name in cls.get_paramname(key)]
        alpha = np.sqrt(1j * values[1] * 2 * np.pi * freq)
        return values[0] / alpha / np.tanh(alpha)

    @staticmethod
    def get_bounds():
        return (0, 2000), (1e-5, 1e4)


class WarburgShort(Component):
    """ defines a semi-infinite Warburg element    """
    @staticmethod
    def get_symbol():
        return 'Ws'

    @staticmethod
    def get_unit():
        return 'Ohm', 's'

    @staticmethod
    def get_paramname(name):
        return name + "_R", name + "_T"

    @classmethod
    def calc(cls, param, key, freq):
        values = [param[name] for name in cls.get_paramname(key)]
        alpha = np.sqrt(1j * values[1] * 2 * np.pi * freq)
        return values[0] / alpha * np.tanh(alpha)

    @staticmethod
    def get_bounds():
        return (0, 2000), (1e2, 1e8)


circuit_components = {key: eval(key) for key in set(globals()) - initial_state
                      if key not in non_element_functions}
