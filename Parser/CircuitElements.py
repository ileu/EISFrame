class Component:
    def __init__(self, key):
        self.key = key

    def get_paramnames(self):
        return self.key

    def __call__(self):
        raise NotImplementedError


class Resistor(Component):
    def __call__(self):
        return self.key


class Capacitor(Component):
    def __call__(self):
        return f"(1j * omega * {self.key}) ** -1"


class CPE(Component):
    def __init__(self, key):
        super().__init__(key)  # not necessary but I guess good code??
        self.key = [self.key + "_0", self.key + "_1"]

    def __call__(self):
        return f"(1j * omega * {self.key[0]}) ** -{self.key[1]}"
