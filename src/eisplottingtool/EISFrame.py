"""
    This Module has been made for automated plotting of EIS data and fitting.
    The main object is called EISFrame which includes most of the features
    Author: Ueli Sauter
    Date last edited: 25.10.2021
    Python Version: 3.9.7
"""
import logging
import os
import re
from typing import TypeVar

import eclabfiles as ecf
import pandas as pd

LOGGER = logging.getLogger(__name__)
T = TypeVar('T', bound='Parent')


def load_df_from_path(path):
    return None


def _get_default_data_param(columns):
    col_names = {}
    for col in columns:
        if match := re.match(r'Ewe[^|]*', col):
            col_names['voltage'] = match.group()
        elif match := re.match(r'I[^|]*', col):
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
        elif match := re.match(r'Ns', col):
            col_names['Ns'] = match.group()
    return col_names


class EISFrame:
    """
       EISFrame used to store the data and plot/fit the data.
    """

    def __init__(
        self,
        name=None,
        path=None,
        df: pd.DataFrame = None,
        **kwargs
    ) -> None:
        """ Initialises an EISFrame

        An EIS frame can plot a Nyquist plot and a lifecycle plot
        of the given cycling data with different default settings.

        Parameters
        ----------
        df : pd.DataFrame

        path: str
            Path to data file

        **kwargs: dict
            circuit: str
                Equivilant circuit for impedance
            cell: Cell
                Battery cell for normalizing
        """
        self.eis_params = kwargs

        if name is not None:
            self.eis_params["name"] = name

        if path is not None:
            self.eis_params["path"] = path

        if df is not None:
            self.df = df

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()

    def __getitem__(self, item):
        if isinstance(item, int):
            self.eis_params['cycle'] = item
            return self
        return self.df.__getitem__(item)

    def load(self, path=None, files=None, cont_time=True):
        if path is None:
            path = self.eis_params["path"]

        if isinstance(files, list):
            end_time = 0
            data = []
            for f in files:
                d = self.load(path + f)
                if cont_time:
                    start_time = d[0].time.iloc[0]
                    for c in d:
                        c.time += end_time - start_time
                    end_time = d[-1].time.iloc[-1]

                data += d

            return data

        if files is not None:
            path = path + files

        ext = os.path.splitext(path)[1][1:]

        if ext in {"csv", 'txt'}:
            data = pd.read_csv(path, sep=',', encoding='unicode_escape')
        elif ext in {'mpr', 'mpt'}:
            data = ecf.to_df(path)
        else:
            raise ValueError(f"Datatype {ext} not supported")

        if data.empty:
            raise ValueError(f"File {path} has no data")

        col_names = _get_default_data_param(data.columns)
        if col_names.get("cycle") is None:
            raise ValueError(f"No cycles detetected in file {path}.")
        print(data.columns)
        print(col_names)
        self.df = data.copy()

        data.set_index([col_names["cycle"], col_names.get("Ns")], inplace=True)
        data.sort_index(inplace=True)

        print(data.columns)
        print(data.loc[1])
        print(data.loc[0, 1])
        print(data.loc[(0, 1), :])
        print(data.loc[(0, 1), col_names["frequency"]])
        print(data.loc[([1, 2, 3], 1), col_names["frequency"]])
        print(data.loc[(slice(None), 1), col_names["frequency"]])
