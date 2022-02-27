import logging
import warnings
from typing import Callable

import numpy as np
from scipy.optimize import least_squares, minimize

logger = logging.getLogger(__name__)


def fit_routine(
    opt_func: Callable[[list], float],
    fit_guess: list[float],
    bounds: list[tuple],
    repeat: int = 1,
):
    """
    Fitting routine which uses scipys least_squares and minimize.

    Least_squares is a good fitting method but will get stuck in local minimas.
    For this reason, the Nelder-Mead-Simplex algorithm is used to get out of these local minima.
    The fitting routine is inspired by Relaxis 3 fitting procedure.
    More information about it can be found on page 188 of revision 1.25 of Relaxis User Manual.
    https://www.rhd-instruments.de/download/manuals/relaxis_manual.pdf

    Open issue is estimate the errors of the parameters. For further information look:
    - https://github.com/andsor/notebooks/blob/master/src/nelder-mead.md
    - https://math.stackexchange.com/questions/2447382/nelder-mead-function-fit-error-estimation-via-surface-fit
    - https://stats.stackexchange.com/questions/424073/calculate-the-uncertainty-of-a-mle

    Parameters
    ----------
    opt_func
        function that gets minimized
    fit_guess
        initial guess for minimization
    bounds
        bounds of the fitting parameters
    repeat
        how many times the least squares and minimize step gets repeated

    Returns
    -------
    opt_result: scipy.optimize.OptimizeResult
        the result of the optimization from the last step of Nelder-Mead.
    """
    initial_value = np.array(fit_guess)

    # least squares have different format for bounds
    ls_bounds_lb = [bound[0] for bound in bounds]
    ls_bounds_ub = [bound[1] for bound in bounds]
    ls_bounds = (ls_bounds_lb, ls_bounds_ub)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in tanh")

        logger.debug(f"Started fitting routine")
        for i in range(repeat):
            logger.debug(f"Fitting routine pass {i}")
            opt_result = least_squares(
                opt_func,
                initial_value,
                bounds=ls_bounds,
                xtol=1e-13,
                max_nfev=1000,
                ftol=1e-9,
            )
            initial_value = opt_result.x
            logger.debug(f"Finished least squares")
            opt_result = minimize(
                opt_func,
                initial_value,
                bounds=bounds,
                tol=1e-13,
                options={"maxiter": 1e4, "fatol": 1e-9},
                method="Nelder-Mead",
            )
            initial_value = opt_result.x
            logger.debug(f"Finished Nelder-Mead")
    logger.debug(f"Finished fitting routine")
    return opt_result
