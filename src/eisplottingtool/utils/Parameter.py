class Parameter:
    """Parameter class to save data of parameters

    A parameter consists of a name, bounds in the form of (ll, ul) with lb =
    lower bounds and ub = upper bounds and a unit string.

    Also used to store fitting results fo the paramater.

    """

    def __init__(self, name, bounds, unit):
        self.name = name
        self.value = 0.0
        self.unit = unit
        self.bounds = bounds
        self.fixed = False

    def __repr__(self):
        name = f"Parameter {self.name}"
        value = rf"{self.value:.3e} [{self.unit}]"
        return f"<{name}, {value}>"

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.name == other.name
        return False
