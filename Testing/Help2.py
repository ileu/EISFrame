import numpy as np
from elements import circuit_elements, get_element_from_name


def wrapCircuit(circuit, constants):
    """ wraps function so we can pass the circuit string """

    def wrappedCircuit(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        circuit : string
        constants : dict
        parameters : list of floats
        frequencies : list of floats

        Returns
        -------
        array of floats

        """

        x = eval(buildCircuit(circuit, frequencies, *parameters,
                              constants=constants, eval_string='',
                              index=0)[0],
                 circuit_elements)
        y_real = np.real(x)
        y_imag = np.imag(x)

        return np.hstack([y_real, y_imag])

    return wrappedCircuit


def wrapCircuit2(circuit, constants):
    """ wraps function so we can pass the circuit string """

    def wrappedCircuit(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        circuit : string
        constants : dict
        parameters : list of floats
        frequencies : list of floats

        Returns
        -------
        array of floats

        """

        x = eval(buildCircuit(circuit, frequencies, *parameters,
                              constants=constants, eval_string='',
                              index=0)[0],
                 circuit_elements)

        return x

    return wrappedCircuit


def check_and_eval(element):
    """ Checks if an element is valid, then evaluates it.

    Parameters
    ----------
    element : str
        Circuit element.

    Raises
    ------
    ValueError
        Raised if an element is not in the list of allowed elements.

    Returns
    -------
    Evaluated element.

    """
    allowed_elements = circuit_elements.keys()
    if element not in allowed_elements:
        raise ValueError(f'{element} not in ' +
                         f'allowed elements ({allowed_elements})')
    else:
        return eval(element, circuit_elements)


def buildCircuit(circuit, frequencies, *parameters,
                 constants=None, eval_string='', index=0):
    """ recursive function that transforms a circuit, parameters, and
    frequencies into a string that can be evaluated

    Parameters
    ----------
    circuit: str
    frequencies: list/tuple/array of floats
    parameters: list/tuple/array of floats
    constants: dict

    Returns
    -------
    eval_string: str
        Python expression for calculating the resulting fit
    index: int
        Tracks parameter index through recursive calling of the function
    """

    parameters = np.array(parameters).tolist()
    frequencies = np.array(frequencies).tolist()
    circuit = circuit.replace(' ', '')

    def parse_circuit(circuit, parallel=False, series=False):
        """ Splits a circuit string by either dashes (series) or commas
            (parallel) outside of any paranthesis. Removes any leading 'p('
            or trailing ')' when in parallel mode """

        assert parallel != series, \
            'Exactly one of parallel or series must be True'

        def count_parens(string):
            return string.count('('), string.count(')')

        if parallel:
            special = ','
            if circuit.endswith(')') and circuit.startswith('p('):
                circuit = circuit[2:-1]
        if series:
            special = '-'

        split = circuit.split(special)

        result = []
        skipped = []
        for i, sub_str in enumerate(split):
            if i not in skipped:
                if '(' not in sub_str and ')' not in sub_str:
                    result.append(sub_str)
                else:
                    open_parens, closed_parens = count_parens(sub_str)
                    if open_parens == closed_parens:
                        result.append(sub_str)
                    else:
                        uneven = True
                        while i < len(split) - 1 and uneven:
                            sub_str += special + split[i + 1]

                            open_parens, closed_parens = count_parens(sub_str)
                            uneven = open_parens != closed_parens

                            i += 1
                            skipped.append(i)
                        result.append(sub_str)
        return result

    parallel = parse_circuit(circuit, parallel=True)
    series = parse_circuit(circuit, series=True)

    if series is not None and len(series) > 1:
        eval_string += "s(["
        split = series
    elif parallel is not None and len(parallel) > 1:
        eval_string += "p(["
        split = parallel
    elif series == parallel:
        eval_string += "(["
        split = series

    for i, elem in enumerate(split):
        if ',' in elem or '-' in elem:
            eval_string, index = buildCircuit(elem, frequencies,
                                              *parameters,
                                              constants=constants,
                                              eval_string=eval_string,
                                              index=index)
        else:
            param_string = ""
            raw_elem = get_element_from_name(elem)
            elem_number = check_and_eval(raw_elem).num_params
            param_list = []
            for j in range(elem_number):
                if elem_number > 1:
                    current_elem = elem + '_{}'.format(j)
                else:
                    current_elem = elem

                if current_elem in constants.keys():
                    param_list.append(constants[current_elem])
                else:
                    param_list.append(parameters[index])
                    index += 1

            param_string += str(param_list)
            new = raw_elem + '(' + param_string + ',' + str(frequencies) + ')'
            eval_string += new

        if i == len(split) - 1:
            eval_string += '])'
        else:
            eval_string += ','

    return eval_string, index
