import logging
import os
import re
import warnings
from typing import Union

import pandas as pd
import eclabfiles as ecf

from eisplottingtool.EISFrame import EISFrame

Logger = logging.getLogger(__name__)


def _get_default_data_param(columns):
    col_names = {}
    for col in columns:
        if match := re.match(r'Ewe[^|]*', col):
            col_names['voltage'] = match.group()
        elif match := re.match(r'I/mA[^|]*', col):
            col_names['current'] = match.group()
        elif match := re.match(r'Re\(Z(we-ce)?\)[^|]*', col):
            col_names['real'] = match.group()
        elif match := re.match(r'-Im\(Z(we-ce)?\)[^|]*', col):
            col_names['imag'] = match.group()
        elif match := re.match(r'Phase\(Z(we-ce)?\)[^|]*', col):
            col_names['phase'] = match.group()
        elif match := re.match(r'\|Z(we-ce)?\|[^|]*', col):
            col_names['abs'] = match.group()
        elif match := re.match(r'time[^|]*', col):
            col_names['time'] = match.group()
        elif match := re.match(r'(z )?cycle( number)?[^|]*', col):
            col_names['cycle'] = match.group()
        elif match := re.match(r'freq[^|]*', col):
            col_names['frequency'] = match.group()
    return col_names


def load_csv_to_df(path: str, sep='\t'):
    return pd.read_csv(path, sep=sep, encoding='unicode_escape')


def load_data(
        path: str,
        file: Union[str, list[str]] = None,
        sep='\t',
        cont_time: bool = True,
        data_param: dict = None,
) -> Union[EISFrame, list[EISFrame]]:
    """
        loads the data from the given path into an EISFrame
    """
    if isinstance(file, list):
        end_time = 0
        data = []
        for f in file:
            d = load_data(path + f, sep=sep, data_param=data_param)
            if cont_time:
                start_time = d[0].time.iloc[0]
                for c in d:
                    c.time += end_time - start_time
                end_time = d[-1].time.iloc[-1]

            data += d

        return data

    if file is not None:
        path = path + file

    __, ext = os.path.splitext(path)

    if ".csv" in path or ".txt" in ext:
        data = load_csv_to_df(path, sep)
    elif ext == '.mpr' or ext == '.mpt':
        data = ecf.to_df(path)
    else:
        warnings.warn("Datatype not supported")
        return []

    # check if data was loaded
    if data.empty:
        warnings.warn("No data was loaded")
        Logger.info("File location: " + path)
        return []

    # check if all the parameters are available
    if data_param is None:
        data_param = _get_default_data_param(data.columns)
    else:
        for param in data_param:
            if param not in data:
                warnings.warn(
                        f"Not valid data file since column {param} is missing"
                )
                print(f"Availible parameters are {data.columns}")
                Logger.debug("File location: " + path)
                return []

    if (cycle_param := data_param.get('cycle')) is None:
        Logger.info("No cycles detected")
        return EISFrame(data, column_names=data_param)

    cycles = []

    min_cyclenumber = int(min(data[cycle_param]))
    max_cyclenumber = int(max(data[cycle_param]))

    for i in range(min_cyclenumber, max_cyclenumber + 1):
        cycle = data[data[cycle_param] == i].reset_index()
        cycles.append(EISFrame(cycle, column_names=data_param))
    return cycles
