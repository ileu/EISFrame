import numpy as np
import pint


class MarkPoint:
    """Special point to mark in an eis plot.

    A mark point is given by a specific frequency. The mark point is described
    by a color and a name. A frequency range can be given to narrow the search
    area in frequency space for a data point.
    """

    def __init__(self, name: str, color: str, freq: float, delta_f: float = -1) -> None:
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


grain_boundaries = MarkPoint("LLZO-GB", "blue", freq=3e5, delta_f=5e4)
hllzo = MarkPoint("HLLZO", "orange", freq=3e4, delta_f=5e3)
lxlzo = MarkPoint("LxLZO", "lime", freq=2e3, delta_f=5e2)
interface = MarkPoint("Interphase", "magenta", freq=50, delta_f=5)
ecr_tail = MarkPoint("ECR", "darkgreen", freq=0.5, delta_f=0.1)

LLZO_mark_points = [grain_boundaries, hllzo, lxlzo, interface, ecr_tail]
