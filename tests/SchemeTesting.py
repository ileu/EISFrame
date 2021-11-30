import schemdraw as sd
import schemdraw.dsp as dsp
from schemdraw import elements as elm


def draw(elements: list, drawing: sd.Drawing):
    drawing += dsp.Line().right().length(drawing.unit / 8.0)
    drawing.move(drawing.unit * 0.75, 0)
    for i in range(len(elements)):
        length = -0.25 * (len(elements) - 1.0) + 0.5 * i
        drawing.push()
        if length >= 0:
            drawing += dsp.Line().up().length(drawing.unit * length)
        else:
            drawing += dsp.Line().down().length(drawing.unit * -length)
        drawing += elements[i].left().length(drawing.unit * 0.75).label('R1')
        if length < 0:
            drawing += dsp.Line().up().length(drawing.unit * -length)
        else:
            drawing += dsp.Line().down().length(drawing.unit * length)
        drawing.pop()

    drawing += dsp.Line().right().length(drawing.unit / 8.0)

    return drawing


def main():
    d = sd.Drawing(fontsize=14)
    d += elm.Resistor().right()
    draw([elm.Resistor(), elm.Resistor(), elm.Resistor(), elm.Resistor()], d)
    draw([elm.Resistor(), elm.Capacitor(), elm.Resistor(), elm.Resistor()], d)
    d += elm.Capacitor().right()

    d.draw()


if __name__ == "__main__":
    main()
