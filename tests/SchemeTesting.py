import numpy as np
import schemdraw as sd
import schemdraw.dsp as dsp
from schemdraw import elements as elm


def draw(elements: list, drawing: sd.Drawing):
    length = len(elements)
    result = []

    for l in np.arange(-length, length, 2) * 0.25 + 0.25:
        print(l)
        drawing.push()
        if l >= 0:
            drawing += dsp.Line().up().length(drawing.unit * l)
        else:
            drawing += dsp.Line().down().length(drawing.unit * -l)
        drawing += elm.Resistor().right()
        if l < 0:
            drawing += dsp.Line().up().length(drawing.unit * -l)
        else:
            drawing += dsp.Line().down().length(drawing.unit * l)
        drawing.pop()
        drawing.draw()

    return result


def main():
    # fig, ax = plt.subplots(1, 1)
    # x = np.linspace(1, 10)
    # ax.plot(x, np.sin(x))
    # elm.style(elm.STYLE_IEC)
    d = sd.Drawing(fontsize=14)
    # d += dsp.Line().length(d.unit / 4)
    # d.push()
    # d += dsp.Line().up().length(d.unit/4)
    # # d += dsp.Line().right().length(d.unit/8)
    # d += elm.Resistor().right().length(d.unit/1.5)
    # d.pop()
    # d += elm.CPE().right().length(d.unit/1.5)
    # d.pop()
    # d += dsp.Line().down().length(d.unit/4)
    # # d += dsp.Line().right().length(d.unit/8)
    # d += elm.Diode().right().length(d.unit/1.5)
    # # d += dsp.Line().right().length(d.unit/8)
    # d += dsp.Line().up().length(d.unit/4)
    # d.push()
    # d += dsp.Line().up().length(d.unit / 4)
    # # d += dsp.Line().left().length(d.unit / 8)
    # d.pop()
    # d += dsp.Line().right().length(d.unit/4)
    # d.draw()

    # d.push()
    # d += dsp.Line().up().length(d.unit / 4)
    # d += elm.Resistor().right()
    # d += dsp.Line().down().length(d.unit / 4)
    # d.pop()
    # d += elm.Resistor()
    draw(['', ''], d)


if __name__ == "__main__":
    main()
