import logging
import warnings

import numpy as np
from scipy.optimize import least_squares, minimize, NonlinearConstraint

LOGGER = logging.getLogger(__name__)


def fit_routine(opt_func, fit_guess, bounds, repeat=1, condition=None):
    """
        Fitting routine which uses scipys least_squares and minimize.

        Least_squares is a good fitting method but will get stuck in local minimas.
        For this reason, the Nelder-Mead-Simplex algorithm is used to get out of
        these local minima.
        The fitting routine is inspired by Relaxis 3 fitting procedure.
        More information about it can be found on page 188 of revison 1.25 of
        Relaxis User Manual.
        https://www.rhd-instruments.de/download/manuals/relaxis_manual.pdf

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
        condition
            if the condition should apply or not

        Returns
        -------
        scipy.optimize.OptimizeResult
            the `OptimizeResult` from the last step
        """
    initial_value = np.array(fit_guess)

    # why does least squares have different format for bounds ???
    ls_bounds_lb = [bound[0] for bound in bounds]
    ls_bounds_ub = [bound[1] for bound in bounds]
    ls_bounds = (ls_bounds_lb, ls_bounds_ub)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in true_divide"
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in true_divide"
        )
        warnings.filterwarnings(
            "ignore",
            message="overflow encountered in tanh"
        )
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in double_scalars"
        )
        warnings.filterwarnings(
            "ignore",
            message="overflow encountered in power"
        )
        LOGGER.debug(f"Started fitting routine with")
        LOGGER.debug(f"Initial guess: {initial_value}")
        LOGGER.debug(f"Bounds: {bounds}")
        LOGGER.debug(f"Least-Squares bounds: {ls_bounds}")

        for i in range(repeat):
            LOGGER.debug(f"Fitting routine pass {i}")
            opt_result = least_squares(
                opt_func,
                initial_value,
                bounds=ls_bounds,
                xtol=1e-13,
                max_nfev=1000,
                ftol=1e-9
            )
            initial_value = opt_result.x
            if condition is None:
                opt_result = minimize(
                    opt_func,
                    initial_value,
                    bounds=bounds,
                    tol=1e-13,
                    options={'maxiter': 1e4, 'ftol': 1e-9},
                    method='Nelder-Mead'
                )
            else:
                nonlin_condition = NonlinearConstraint(
                    condition,
                    -50,
                    50
                )
                opt_result = minimize(
                    opt_func,
                    initial_value,
                    bounds=bounds,
                    tol=1e-13,
                    options={'maxiter': 1e4, 'gtol': 1e-9},
                    # method='Nelder-Mead'
                    method='trust-constr',
                    constraints=nonlin_condition
                )
            initial_value = opt_result.x
    LOGGER.debug(f"Finished fitting routine")
    return opt_result
