# eisplottingtools

This package was created during my civil service at EMPA. to process, plot and fit electrochemical impedance
spectroscopy (EIS) data.

This package itself centered around the class `EISFrame`, which itself is a wrapper around
a [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). The `EISFrame` contains
additional methods to plot, fit and manipulate EIS data and also other electrochemical data.

## Installation

This package is currently not on [PyPI](https://pypi.org/).  
To install and use this package fork or clone the repository on [GitHub](https://github.com/ileu/eisplottingtool). Move
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

To load data and to create an `EISFrame`the `load_data`function can be used.  
The following file types are supported to load data from: `csv`, `txt`, `mpr`, `mpt`.

```python
import eisplottingtool as ept

data = ept.load_data("Example_Data.mpr")
```

### Step 2: Manipulate Data

An `EISFrame` is essentially a wrapper around a `pandas.DataFrame` with additional functionality.  
Like a `DataFrame` the data in an `EisFrame` can be accessed directly with the index operator `[...]`.  
- int for accessing cycles
- 


### Step 3: Define an equivalent circuit

To fit our data an equivalent circuit is needed, defined as a string.  
To combine Elements  are combined in series by a dash (`-`) is used. Elements in parallel are wrapped by `p( , )`.  
An element is defined by a symbol made of letters followed by a digit. Implemented components are:

For example the following circuit would be defined as:

#### Implemented components

| Name                          | Symbol | Parameters | Default Bounds | Units        |
|-------------------------------|--------|------------|----------------|--------------|
| Resistor                      | R      | R          | (1e-6, 1e6)    | Ohm          |
| Capacitance                   | C      | C          | (1e-20, 1)     | Farad        |
| Constant Phase Element        | CPE    | CPE_Q      | (1e-20, 1)     | Ohm^-1 s^n   |
|                               |        | CPE_n      | (0, 1)         |              |
| Warburg element               | W      | W          | (0, 1e10)      | Ohm^-1 s^0.5 |
| Warburg short element         | Ws     | Ws_R       | (0, 1e10)      | Ohm          |
|                               |        | Ws_T       | (1e-10, 1e10)  | s            |
| Warburg open element          | Wo     | Wo_R       | (0, 1e10)      | Ohm          |
|                               |        | Wo_T       | (1e-10, 1e10)  | s            |
| Warburg short element special | Wss    | Wss_R      | (0, 1e10)      | Ohm          |
|                               |        | Wss_T      | (1e-10, 1e10)  | s            |
|                               |        | Wss_n      | (0, 1)         |              |
| Warburg open element special  | Wos    | Wos_R      | (0, 1e10)      | Ohm          |
|                               |        | Wos_T      | (1e-10, 1e10)  | s            |
|                               |        | Wos_n      | (0, 1)         |              |

### Step 4: Fit the data

Least_squares is a good fitting method but will get stuck in local minima. For this reason, the Nelder-Mead-Simplex
algorithm is used to get out of these local minima. The fitting routine is inspired by Relaxis 3 fitting procedure. More
information about it can be found on page 188 of revision 1.25
of [Relaxis User Manual](https://www.rhd-instruments.de/download/manuals/relaxis_manual.pdf).

### Step 5: visualize the data

## Functions

Extended Backus–Naur form (EBNF) parser for a circuit string.

Implements an extended Backus–Naur form (EBNF)  to parse a string that describes a circuit Already implemented circuit
elements are located in CircuitComponents.py To use a component in the circuit string use its symbol. The symbol can be
followed by a digit to differentiate similar components.

To put elements in series connect them through '-' Parallel elements are created by p(Elm1, Elm2,...)

The syntax of the EBNF is given by:
- circuit = element | element-circuit - element = component | parallel - parallel = p(circuit {,circuit})
- component = a circuit component defined in ``circuit_components``

From this an equation for the impedance is generated and evaluated.
