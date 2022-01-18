"""
    This Module has been made for automated plotting of EIS data and fitting.
    The main object is called EISFrame which includes most of the features
    Author: Ueli Sauter
    Date last edited: 25.10.2021
    Python Version: 3.9.7
"""
import logging
from typing import TypeVar

import pandas as pd

LOGGER = logging.getLogger(__name__)
T = TypeVar('T', bound='Parent')


def load_df_from_path(path):
    return None


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
