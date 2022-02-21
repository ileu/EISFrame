# eisplottingtools

This package was created during my civil service at EMPA. to process, plot and fit electrochemical impedance
spectroscopy (EIS) data.

This package itself centered around the class `EISFrame`, which itself is a wrapper around a [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
The `EISFrame` contains additional methods to plot, fit and manipule EIS data and also other electrochemical data.

## Installation

This package is currently not on [PyPI](https://pypi.org/).  
To install and use this package fork or clone the repository on [Github](https://github.com/ileu/eisplottingtool). Move
to the folder where the repository is located on your local machine and open a Terminal.

To install run:

```bash
> py -m pip install .
```

to install and editable version use:

```bash
> py -m pip install -e .
```

## Example Usage


### Step 1: Import data & create EISFrame
To load data and to create an `EISFrame` the [``load_data``](https://github.com/ileu/eisplottingtool/blob/a11aa63c97564c4ce0d4ca72a06c03ac0ccefbf6/src/eisplottingtool/loading.py#L45) function can be used
The function supports the following file types: `csv`, `txt`, `mpr`, `mpt`.

```python
import eisplottingtool as ept
data = ept.load_data("Example_Data.mpr")
```

### Step 2: Manipulate Data
TODO

### Step 3: Define an equivalant circuit
To define ean equivalent circuit or impedance model


For the fitting a model or equivilant circuit is needed. The equivilant circuit is defined as a string.
To combine elements in series a dash (-) is used. Elements in parallel are wrapped by p( , ).
An element is definied by an identifier (usually letters) followed by a digit.
Already implemented elements are located in :class:`circuit_components<circuit_utils.circuit_components>`:

#### Implemented components
The already implemented companetns are:

| Name                          | Symbol | Paramters | Bounds        | Units        |
|-------------------------------|--------|-----------|---------------|--------------|
| Resistor                      | R      | R         | (1e-6, 1e6)   | Ohm          |
| Capacitance                   | C      | C         | (1e-20, 1)    | Farrad       |
| Constant Phase Element        | CPE    | CPE_Q     | (1e-20, 1)    | Ohm^-1 s^n   |
|                               |        | CPE_n     | (0, 1)        |              |
| Warburg element               | W      | W         | (0, 1e10)     | Ohm^-1 s^0.5 |
| Warburg short element         | Ws     | Ws_R      | (0, 1e10)     | Ohm          |
|                               |        | Ws_T      | (1e-10, 1e10) | s            |
| Warburg open elemnt           | Wo     | Wo_R      | (0, 1e10)     | Ohm          |
|                               |        | Wo_T      | (1e-10, 1e10) | s            |
| Warburg short element special | Wss    | Wss_R     | (0, 1e10)     | Ohm          |
|                               |        | Wss_T     | (1e-10, 1e10) | s            |
|                               |        | Wss_n     | (0, 1)        |              |
| Warburg open elemnt special   | Wos    | Wos_R     | (0, 1e10)     | Ohm          |
|                               |        | Wos_T     | (1e-10, 1e10) | s            |
|                               |        | Wos_n     | (0, 1)        |              |

### Step 4: Fit the data

Least_squares is a good fitting method but will get stuck in local minimas.
For this reason, the Nelder-Mead-Simplex algorithm is used to get out of these local minima.
The fitting routine is inspired by Relaxis 3 fitting procedure.
More information about it can be found on page 188 of revison 1.25 of [Relaxis User Manual](https://www.rhd-instruments.de/download/manuals/relaxis_manual.pdf).

### Step 5: plot the data

## Functions

Extended Backus–Naur form (EBNF) parser for a circuit string.

Implements an extended Backus–Naur form (EBNF)  to parse a string that descirbes a circuit
Already implemented circuit elements are locacted in CircuitComponents.py
To use a component in the circuit string use its symbol. The symbol can be followed
by a digit to differentiate similar components.

To put elements in series connect them through '-'
Parallel elements are created by p(Elm1, Elm2,...)

The syntax of the EBNF is given by:
    - circuit = element | element-circuit
    - element = component | parallel
    - parallel = p(circuit {,circuit})
    - component = a circuit component defined in ``circuit_components``

From this an equation for the impedance is generated and evaluated.
