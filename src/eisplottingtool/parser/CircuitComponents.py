from typing import Type

import numpy as np
from schemdraw import elements as elm

import eisplottingtool.parser.SchemeElements as sElm
from eisplottingtool.utils.Parameter import Parameter


class Component:
    """ A component which can be used in a circuit string

    A component stores information
    """

    @staticmethod
    def get_symbol() -> str:
        raise NotImplementedError

    @staticmethod
    def get_parameters(name: str) -> list[Parameter]:
        raise NotImplementedError

    @staticmethod
    def calc(param: dict, key: str, freq: np.array) -> np.array:
        raise NotImplementedError

    @staticmethod
    def draw() -> elm.Element:
        return elm.ResistorIEC()

    def __repr__(self):
        return f"Component {self.get_symbol()}"


class Resistor(Component):
    @staticmethod
    def get_symbol():
        return 'R'

    @staticmethod
    def get_parameters(name):
        param = Parameter(name, (1e-6, 2000), 'Ohm')
        return [param]

    @staticmethod
    def calc(param, key, freq):
        result = np.full_like(freq, param.get(key), dtype=float)
        return result


class Capacitor(Component):
    @staticmethod
    def get_symbol():
        return 'C'

    @staticmethod
    def get_parameters(name):
        param = Parameter(name, (1e-15, 1), 'F')
        return [param]

    @staticmethod
    def calc(param, key, freq):
        value = param.get(key)
        result = 1.0 / (1j * 2 * np.pi * freq * value)
        return np.array(result)

    @staticmethod
    def draw():
        return elm.Capacitor()


class CPE(Component):
    @staticmethod
    def get_symbol():
        return 'CPE'

    @staticmethod
    def get_parameters(name):
        param1 = Parameter(name + "_Q", (1e-15, 1), 'Ohm^-1 s^n')
        param2 = Parameter(name + "_n", (0, 1), '')
        return [param1, param2]

    @staticmethod
    def calc(param, key, freq):
        q = param.get(key + "_Q")
        n = param.get(key + "_n")
        result = (1j * 2 * np.pi * freq) ** (-n) / q
        return np.array(result)

    @staticmethod
    def draw():
        return sElm.CPE()


class Warburg(Component):
    """ defines a semi-infinite Warburg element    """

    @staticmethod
    def get_symbol():
        return 'W'

    @staticmethod
    def get_parameters(name):
        param = Parameter(name, (0, 2000), 'Ohm^-1 s^-1/2')
        return [param]

    @staticmethod
    def calc(param, key, freq):
        value = param.get(key)
        result = value * (1 - 1j) / np.sqrt(2 * np.pi * freq)
        return np.array(result)

    @staticmethod
    def draw():
        return sElm.Warburg()


class WarburgOpen(Component):
    """ defines a finite-space Warburg element    """

    @staticmethod
    def get_symbol():
        return 'Wo'

    @staticmethod
    def get_parameters(name):
        param1 = Parameter(name + "_R", (0, 2000), 'Ohm')
        param2 = Parameter(name + "_T", (1e-5, 1e4), 's')
        return [param1, param2]

    @staticmethod
    def calc(param, key, freq):
        r = param.get(key + "_R")
        t = param.get(key + "_T")
        alpha = np.sqrt(1j * t * 2 * np.pi * freq)
        result = r / alpha / np.tanh(alpha)
        return np.array(result)

    @staticmethod
    def draw():
        return sElm.WarburgOpen()


class WarburgShort(Component):
    """ defines a finite-length Warburg element    """

    @staticmethod
    def get_symbol():
        return 'Ws'

    @staticmethod
    def get_parameters(name):
        param1 = Parameter(name + "_R", (0, 2000), 'Ohm')
        param2 = Parameter(name + "_T", (1e-3, 1e8), 's')
        return [param1, param2]

    @staticmethod
    def calc(param, key, freq):
        r = param.get(key + "_R")
        t = param.get(key + "_T")
        alpha = np.sqrt(1j * t * 2 * np.pi * freq)
        result = r / alpha * np.tanh(alpha)
        return np.array(result)

    @staticmethod
    def draw():
        return sElm.WarburgShort()


class WarburgShortSpezial(Component):
    """ defines a finite-length Warburg element with variable exponent """

    @staticmethod
    def get_symbol():
        return 'Wss'

    @staticmethod
    def get_parameters(name):
        param1 = Parameter(name + "_R", (0, 2000), 'Ohm')
        param2 = Parameter(name + "_T", (1e-3, 1e8), 's')
        param3 = Parameter(name + "_n", (0, 0.5), '')
        return [param1, param2, param3]

    @staticmethod
    def calc(param, key, freq):
        r = param.get(key + "_R")
        t = param.get(key + "_T")
        n = param.get(key + "_n")
        alpha = np.power(1j * t * 2 * np.pi * freq, n)
        result = r / alpha * np.tanh(alpha)
        return np.array(result)

    @staticmethod
    def draw():
        return sElm.WarburgShort()


circuit_components: dict[str, Type[Component]] = {
    'Resistor': Resistor,
    'Capacitor': Capacitor,
    'CPE': CPE,
    'Warburg': Warburg,
    'WarburgShort': WarburgShort,
    'WarburgOpen': WarburgOpen,
    'WarburgShortSpecial': WarburgShortSpezial,
}
