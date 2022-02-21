# eisplottingtools

This package was created during my civil service at EMPA. to process, plot and fit electrochemical impedance
spectroscopy (EIS) data.

This is a still TODO. The source code is on .

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

### Step 3: Define an equivalant circuit
To define ean equivalent circuit or impedance model


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
