import numpy as np


class Parameter:
    """ Parameter class to save data of parameters

    A parameter consists of a name, bounds in the form of (ll, ul) with lb =
    lower bounds and ub = upper bounds and a unit string.

    Also used to store fitting results fo the paramater.

    """

    def __init__(self, name, bounds, unit):
        self.name = name
        self.value = 0.0
        self.error = 0.0
        self.unit = unit
        self.bounds = bounds

    def __repr__(self):
        name = f"Parameter {self.name}"
        value = rf"{self.value} ($\pm${self.error}) [{self.unit}]"
        return f"{name}, {value}"

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.name == other.name
        return False


class ParameterDict(list):
    def __init__(self, iterable=None, *args):
        super().__init__()
        if iterable:
            for item in iterable:
                self.append(item)

        for arg in args:
            self.append(arg)

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)
        for item in self:
            if item.name == key:
                return item
        return None

    def __repr__(self):
        return super().__repr__()


class Cell:
    """
        Save the characteristics of a cell. Usefull for further calculation.
    """

    def __init__(self, diameter_mm, thickness_mm):
        """
         Initializer of a cell

        Parameters
        ----------
        diameter_mm : float
            diameter of the cell in mm
        thickness_mm height : float
            thickness of the cell in mm
        """
        self.diameter_mm = diameter_mm
        self.height_mm = thickness_mm

        self.area_mm2 = (diameter_mm / 2) ** 2 * np.pi  # area of the cell
        self.volume_mm3 = self.area_mm2 * thickness_mm  # volume of the cell

    def __repr__(self):
        return f"{self.diameter_mm=}, {self.height_mm=}, {self.area_mm2=}"

    def __str__(self):
        return "Cell with " + self.__repr__()
