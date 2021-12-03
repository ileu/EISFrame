import re
from typing import Callable

import matplotlib.axes
import schemdraw as sd
from schemdraw import dsp

from eisplottingtool.parser import circuit_components
from eisplottingtool.parser.CircuitComponents import Parameter


def parse_circuit3(
        circ: str,
        draw: bool = False
        ) -> tuple[list[Parameter], Callable]:
    """ EBNF parser for a circuit string.

    Implements an extended Backus–Naur form to parse a string that descirbes
    a circuit.
    Already implemented circuit elements are locacted in CircuitComponents.py

    To put elements in series connect them through -.
    Parallel elements are created by p(Elm1, Elm2,...)

    The syntax of the EBNF is given by:

    circuit = element | element-circuit
    element = component | parallel
    parallel = p(circuit {,circuit})
    component = a circuit component

    Parameters
    ----------
    circ : str
        String that descirbes a circuit
    draw : bool
        If the circuit should be drawn. Default False.

    Returns
    -------
    param_info : dict
    calculate : Callable

    """
    param_info: list[Parameter] = []
    draw_list = []

    def component(c: str):
        """ process component and remove from circuit string c

        Parameters
        ----------
        c : str
            circuit string

        Returns
        -------

        """
        index = re.match(r'([a-zA-Z]+)_?\d?', c)
        name = c[:index.end()]
        c = c[index.end():]
        symbol = re.match('[A-Za-z]+', name).group()

        for key, comp in circuit_components.items():
            if comp.get_symbol() == symbol:
                break
        else:
            return c, 1

        param_info.extend(comp.get_parameters(name))
        return c, key + rf".calc(param,'{name}', omega)", comp

    def parallel(c: str):
        c = c[2:]
        tot_eq = ''
        comp_list = []
        while not c.startswith(')'):
            if c.startswith(','):
                c = c[1:]
            c, eq, comp = circuit(c)
            comp_list.append(comp)
            if tot_eq:
                tot_eq += " + "
            tot_eq += f"1.0 / ({eq})"
        c = c[1:]
        return c, f"1.0 / ({tot_eq})", comp_list

    def element(c: str):
        if c.startswith('p('):
            c, eq, comps = parallel(c)
        else:
            c, eq, comps = component(c)
        return c, eq, comps

    def circuit(c: str):
        if not c:
            return c, ''
        c, eq, comps = element(c)
        tot_eq = f"{eq}"
        comp_list = [comps]
        if c.startswith('-'):
            c, eq, comps = circuit(c[1:])
            tot_eq += f" + {eq}"
            comp_list.extend(comps)
        return c, tot_eq, comp_list

    __, equation, drawing = circuit(circ.replace(" ", ""))

    calculate = eval('lambda param, omega: ' + equation, circuit_components)
    return param_info, calculate, drawing


def F(n):
    import numpy as np
    return ((1+np.sqrt(5))**n-(1-np.sqrt(5))**n)/(2**n*np.sqrt(5))


def draw_circuit(circuit, drawing, scale_h=0.25, scale_e=1.0):
    unit = drawing.unit
    scale_factor = 1.0 / F(scale_e)
    for circ in circuit:
        if isinstance(circ, list):
            n = len(circ)
            print(f"parallel {n=}")
            max_length = max(len(element) for element in circ)
            drawing += dsp.Line().right().length(0.25).color('brown')
            for i, d in enumerate(circ):
                print(f"{max_length=}, {len(circ)=}")
                height = -(n - 1) * scale_h + 2 * scale_h * i
                drawing.push()
                if height > 0:
                    drawing += dsp.Line().up().length(unit * height).color('red')
                elif height < 0:
                    drawing += dsp.Line().down().length(unit * -height).color('red')

                if (length := max_length - len(d)) > 0:
                    drawing += dsp.Line().right().length(length * unit * 0.5).color('magenta')

                if isinstance(d, list):
                    n_inner = len(d)
                    print(f"\t inside {n_inner=}, {height=}")
                    draw_circuit(d, drawing, scale_h=0.5 * scale_h, scale_e=scale_e+1)
                else:
                    print("series 2")
                    drawing += d().draw().right().color('red')

                if (length := max_length - len(d)) > 0:
                    drawing += dsp.Line().right().length(length * unit * 0.5).color('magenta')

                if height > 0:
                    drawing += dsp.Line().down().length(unit * height).color('blue')
                elif height < 0:
                    drawing += dsp.Line().up().length(unit * -height).color('blue')
                # if not i == (len(circ)-1):
                drawing.pop()
            drawing.move(max_length * unit-0.5, 0)
            drawing += dsp.Line().right().length(0.25).color('orange')


        else:
            print("series 1")
            drawing += dsp.Line().right().scale((1-scale_factor) / 2.0).color('cyan').length(unit - 0.5)
            drawing += circ().draw().right().color('green').scale(scale_factor).label(str(int(1.0 / scale_factor))).length(unit - 0.5)
            drawing += dsp.Line().right().scale((1-scale_factor) / 2.0).color('cyan').length(unit - 0.5)


def main():

    import numpy as np
    from matplotlib import pyplot as plt
    matplotlib.use('TkAgg')
    circuit = 'R-p(C,Ws-p(R,R))-R'
    circuit2 = 'R-p(CPE,CPE)-p(R-R,Ws-p(R,R-p(R,R)),R)-p(C,C,C,C)-R'
    info, calc, drawing = parse_circuit3(circuit2, draw=True)

    test = True
    d = sd.Drawing(unit=2)

    if test:
        draw_circuit(drawing, d)
    else:
        import schemdraw.elements as elm
        d.move(0, 0.5)
        d.push()
        d += elm.Resistor().color('green')
        d += elm.Resistor().color('red')
        d += elm.Resistor().color('blue')
        d.pop()
        d.move(0, -1)
        d += elm.Resistor().color('green')
        d += dsp.Line().scale(0.9 / 2.0)
        d += elm.Resistor().scale(0.1).color('red')
        d += dsp.Line().scale(0.9 / 2.0)
        d += elm.Resistor().color('blue')

    test = d.draw(showframe=True, show=False)
    ax: matplotlib.axes.Axes = test.ax
    ax.locator_params(axis='x', nbins=20)
    x_ticks = np.array(ax.get_xlim()) * 1.1
    y_ticks = np.array(ax.get_ylim()) * 1.1
    ax.set_xticks(np.arange(*np.round(x_ticks), 0.5))
    ax.set_xticks(np.arange(*np.round(x_ticks), 0.5) + 0.25, minor=True)
    ax.set_yticks(np.arange(*np.round(y_ticks), 0.25))
    ax.grid(which='both')
    plt.show()


if __name__ == "__main__":
    main()
