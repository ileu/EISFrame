initial_state = set(globals().copy())
non_element_functions = ['Component', 'initial_state', 'non_element_functions',
                         'Circuit']


# TODO: add default bounds


class Circuit:
    def __init__(self, circ_string):
        self.circ_string = circ_string


class Component:
    def __init__(self, key):
        self.key = key

    def get_paramnames(self):
        return self.key

    def __call__(self):
        raise NotImplementedError

    @staticmethod
    def get_symbol():
        raise NotImplementedError

    @staticmethod
    def get_unit():
        raise NotImplementedError


class Resistor(Component):
    def __call__(self):
        return self.key

    @staticmethod
    def get_symbol():
        return 'R'

    @staticmethod
    def get_unit():
        return 'Ohm'


class Capacitor(Component):
    def __call__(self):
        return f"1.0 / (1j * omega * {self.key})"

    @staticmethod
    def get_symbol():
        return 'C'

    @staticmethod
    def get_unit():
        return 'F'


class CPE(Component):
    def __init__(self, key):
        super().__init__(key)  # not necessary but I guess good code??
        self.key = [self.key + "_0", self.key + "_1"]

    def __call__(self):
        return f"(1j * omega * {self.key[0]}) ** -{self.key[1]}"

    @staticmethod
    def get_symbol():
        return 'CPE'

    @staticmethod
    def get_unit():
        return 'Ohm^-1 sec^a'


class Warburg(Component):
    """ defines a semi-infinite Warburg element    """

    def __call__(self):
        return f"{self.key} * (1 - 1j) / np.sqrt(omega)"

    @staticmethod
    def get_symbol():
        return 'W'

    @staticmethod
    def get_unit():
        return 'Ohm sec^-1/2'


class WarburgOpen(Component):
    """ defines a semi-infinite Warburg element    """

    def __init__(self, key):
        super().__init__(key)  # not necessary but I guess good code??
        self.key = [self.key + "_0", self.key + "_1"]

    def __call__(self):
        return f"{self.key[0]} / np.sqrt(1j * omega) / np.tanh({self.key[1]} * np.sqrt(1j * omega))"

    @staticmethod
    def get_symbol():
        return 'Wo'

    @staticmethod
    def get_unit():
        return 'Ohm sec^-1/2'


class WarburgShort(Component):
    """ defines a semi-infinite Warburg element    """

    def __init__(self, key):
        super().__init__(key)  # not necessary but I guess good code??
        self.key = [self.key + "_0", self.key + "_1"]

    def __call__(self):
        return f"{self.key[0]} / np.sqrt(1j * omega) * np.tanh({self.key[1]} * np.sqrt(1j * omega))"

    @staticmethod
    def get_symbol():
        return 'Ws'

    @staticmethod
    def get_unit():
        return 'Ohm sec^-1/2'


circuit_components = {key: eval(key) for key in set(globals()) - initial_state
                      if key not in non_element_functions}
