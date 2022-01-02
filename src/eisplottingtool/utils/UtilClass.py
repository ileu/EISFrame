import numpy as np
import pint


class Parameter:
    """ Parameter class to save data of parameters

    A parameter consists of a name, bounds in the form of (ll, ul) with lb =
    lower bounds and ub = upper bounds and a unit string.

    Also used to store fitting results fo the paramater.

    """

    def __init__(self, name, bounds, unit):
        self.name = name
        self.color = 'black'
        self.value = 0.0
        self.error = 0.0
        self.unit = unit
        self.bounds = bounds

    def __repr__(self):
        name = f"Parameter {self.name}, {self.color}"
        value = rf"{self.value} ($\pm${self.error}) [{self.unit}]"
        return f"{name}, {value}"

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.name == other.name
        return False


class MarkPoint:
    """ Special point to mark in an eis plot.

    A mark point is given by a specific frequency. The mark point is described
    by a color and a name. A frequency range can be given to narrow the search
    area in frequency space for a data point.
    """

    def __init__(
            self,
            name: str,
            color: str,
            freq: float,
            delta_f: float = -1
            ) -> None:
        """
        Special point in the EIS spectrum

        Parameters
        ----------
        name : str
            Name of the mark point
        color : str
            Color of the mark point
        freq : float
         Specific frequency of the feature
        delta_f : float
            interval to look for datapoints, defualt is 10% of freq
        """
        self.name = name
        self.color = color
        self.freq = freq
        if delta_f <= 0:
            self.delta_f = freq // 10
        else:
            self.delta_f = delta_f
        self.index = -1  # index of the first found data point matching in

        self.left = self.freq - self.delta_f  # left border of freq range
        self.right = self.freq + self.delta_f  # right border of freq range

        self.magnitude = np.floor(np.log10(freq))  # magnitude of the frequency

    def __repr__(self):
        out = f"{self.name} @ {self.freq} (1e{self.magnitude}), "
        out += f"{self.color=} "
        out += f"with {self.index=}"
        return out

    def label(self, freq=None):
        ureg = pint.UnitRegistry()
        if freq is None:
            f = self.freq
        else:
            f = freq[self.index]
        label = f * ureg.Hz
        return f"{label.to_compact():~.0f}"


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


grain_boundaries = MarkPoint('LLZO-GB', 'blue', freq=3e5, delta_f=5e4)
hllzo = MarkPoint('HLLZO', 'orange', freq=3e4, delta_f=5e3)
lxlzo = MarkPoint('LxLZO', 'lime', freq=2e3, delta_f=5e2)
interface = MarkPoint('Interphase', 'magenta', freq=50, delta_f=5)
ecr_tail = MarkPoint('ECR', 'darkgreen', freq=0.5, delta_f=1)

default_mark_points = [grain_boundaries, hllzo, lxlzo, interface, ecr_tail]
