import math

from schemdraw import Segment, SegmentText
from schemdraw.elements import Element2Term

gap = (math.nan, math.nan)
height = 0.25
width = 1.0
text_size = 20


class CPE(Element2Term):
    """ Constant Phase Element """

    def __init__(self, *d, **kwargs):
        super().__init__(*d, **kwargs)
        capgap = 0.25
        offset = 0.5
        self.segments.append(
                Segment(
                        [(0, 0), (offset, 0), (capgap, -height), gap, (offset, 0),
                         gap, (offset + capgap, 0), (2 * offset, 0)]
                )
        )
        self.segments.append(
                Segment(
                        [(offset, 0), (capgap, height), gap, (offset, 0), gap,
                         (offset + capgap, 0)]
                )
        )
        self.segments.append(
                Segment(
                        [(offset + capgap, 0), (offset, height), gap,
                         (offset + capgap, 0)]
                )
        )
        self.segments.append(
                Segment(
                        [(offset + capgap, 0), (offset, -height), gap,
                         (offset + capgap, 0)]
                )
        )


class Warburg(Element2Term):
    """ Warburg element """

    def __init__(self, *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments.append(
                Segment(
                        [(0, 0), (0, height), (width, height), (width, -height),
                         (0, -height), (0, 0), gap, (width, 0)]
                )
        )
        self.segments.append(
                SegmentText(
                        (width * 0.5, 0),
                        'W',
                        fontsize=text_size)
        )


class WarburgOpen(Element2Term):
    """  Open Warburg element """

    def __init__(self, *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments.append(
                Segment(
                        [(0, 0), (0, height), (width, height), (width, -height),
                         (0, -height), (0, 0), gap, (width, 0)]
                )
        )
        self.segments.append(
                SegmentText(
                        (width * 0.5, 0),
                        'Wo',
                        fontsize=text_size)
        )


class WarburgShort(Element2Term):
    """ Short Warburg element """

    def __init__(self, *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments.append(
                Segment(
                        [(0, 0), (0, height), (width, height), (width, -height),
                         (0, -height), (0, 0), gap, (width, 0)]
                )
        )
        self.segments.append(
                SegmentText(
                        (width * 0.5, 0),
                        'Ws',
                        fontsize=text_size
                )
        )
