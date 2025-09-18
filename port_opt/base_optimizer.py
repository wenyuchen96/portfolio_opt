import collections
import copy
import json
import warnings
from collections.abc import Iterable
from typing import List, Callable

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.optimize as sco

from port_opt import exceptions, objective_functions

class BaseOptimizer:
    """
    Instance variables:
    - 'n_assets': number of assets
    - 'tickers': list of tickers
    - 'weights': asset weights in the portfolio

    Public methods:
    - 'set_weights()' creates self.weights (np.ndarray) from a weights dict
    - 'clean_weights()' rounds the weights and clips near-zeros
    - 'save_weights_to_file()' saves the weights to csv, json, or txt
    """

    def __init__(self, n_assets: int, tickers: List[str] = None):
        self.n_assets = n_assets

        if tickers is None:
            self.tickers = list(range(n_assets))
        else:
            self.tickers = tickers

        self._risk_free_rate = None
        self.weights = None
    
    def _make_output_weights(self, weights: np.ndarray = None):
        """
        Utility function to make output weight dict from weight attribute. If no arguments passed, use self.tickers and self.weights. If one argument is passed, assume it is an alternative weight array so use self.tickers and the argument.
        """
        if weights is None:
            weights = self.weights
        
        #Convert numpy float64 to plain Python float
        weights = [float(w) for w in weights]
        return collections.OrderedDict(zip(self.tickers, weights))
    
    def set_weights(self, input_weights: dict):
        """
        Utility function to set weights attribute from user input weights.

        input_weights: {ticker: weight} dict
        """
        self.weights = np.array([input_weights[ticker] for ticker in self.tickers])

    def clean_weights(self, cutoff:float = 1e-4, rounding:int = 5):
        """
        Helper method to clean the raw weights, setting any weights below the cutoff to zero and rounding the rest.
        
        cutoff: the cutoff value for the weights, default to 1e-4.
        rounding: the number of decimal places to round the weights to, default to 5.
        """
        
        if self.clean_weights is None:
            raise AttributeError("Weights not yet computed")
        clean_weights = self.weights.copy()
        clean_weights[np.abs(clean_weights) < cutoff] = 0
        if rounding is not None:
            if not isinstance(rounding, int) or rounding < 1:
                raise ValueError("rounding must be a positive integer")
            clean_weights = np.round(clean_weights, rounding)

        return self._make_output_weights(clean_weights)
    
    def save_weights_to_file(self, filename: str = "weights.csv"):
        """
        Save the weights to a text file.
        
        filename: name of file. Should be csv, json, or txt.
        """

        clean_weights = self.clean_weights()

        ext = filename.split(".")[-1].lower()
        if ext == "csv":
            pd.Series(clean_weights).to_csv(filename, header=False)
        elif ext == "json":
            with open(filename, "w") as fp:
                json.dump(clean_weights, fp)
        elif ext == "txt":
            with open(filename, "w") as f:
                f.write(str(dict(clean_weights)))
        else:
            raise NotImplementedError("Only supports .txt .json .csv")
        
class BaseConvexOptimizer(BaseOptimizer):
    """
    BaseConvexOptimizer contains many private variables for use by 'cvxpy'.

    Instance variables:
    - 'n_assets': number of assets
    - 'tickers': list of tickers
    - 'weights': asset weights in the portfolio
    - '_opt': cp.Problem
    - '_solver'
    - '_solver_options'

    Public methods:
    -'add_objective()': adds a convex objective to the optimization problem
    -'add_constraint()': adds a constraint to the optimization problem
    -'convex_objective()': solves for a generic convex objective with linear constraints
    -'nonconvex_objective()': solves for a generic non-convex objective using scipy backend
    -'set_weights()': creates self.weights (np.ndarray) from a weights dict
    -'clean_weights()': rounds the weights and clips near-zeros
    -'save_weights_to_file()': saves the weights to csv, json, or txt
    """

    def __init__(
        self,
        n_assets: int,
        tickers: List[str] = None,
        weight_bounds: tuple | tuple[float, float] = (0,1),
        solver: str = None,
        verbose: bool = False,
        solver_options: dict = None,
    ):
        super().__init__(n_assets, tickers)

        #optimization variables
        self._w = cp.Variable(n_assets)
        self._objective = None
        self._additional_objectives = []
        self._constraints = []
        self._lower_bounds = None
        self._upper_bounds = None
        self._opt = None
        self._solver = solver
        self._verbose = verbose
        self._solver_options = solver_options if solver_options else {}
        self._map_bounds_to_constraints(weight_bounds)

    def deepcopy(self):
        """
        returns a custom deep copy of the optimizer, as cvxpy does not support deepcopy
        """
        self_copy = copy.copy(self)
        self_copy._additional_objectives = [
            copy.copy(obj) for obj in self._additional_objectives        
        ]
        self_copy._constraints = [
            copy.copy(constraint) for constraint in self._constraints
        ]
        return self._copy
    
    def _map_bounds_to_constraints(self, test_bounds: tuple | tuple[float, float]):
        """
        Convert input bound into a form acceptable bt cvxpy and add to the constraints list.

        test_bounds: minimum and maximum weight of each asset OR single min/max pair if all identical OR pairs of arrays corresponding to lower/upper bounds; default to (0,1)
        """

        if len(test_bounds) == self.n_assets and not isinstance(test_bounds[0], (float, int)):
            bounds = np.array(test_bounds, dtype=float)
            self._lower_bounds = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            self._upper_bounds = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            #otherwise this must be a pair
            if len(test_bounds) != 2 or not isinstance(test_bounds, (tuple, list)):
                raise TypeError(
                    "test_bounds must be a pair (lower bound, upper bound) OR a collection of bounds for each asset"
                )
            lower, upper = test_bounds

            #replace None values with appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = -1 if lower is None else lower
                self._lower_bounds = np.array([lower] * self.n_assets)
                upper = 1 if upper is None else upper
                self._upper_bounds = np.array([upper] * self.n_assets)
            else: #1 array represents all the lower bounds, and 1 array represents all the upper bounds
                self._lower_bounds = np.nan_to_num(lower, nan=-1)
                self._upper_bounds = np.nan_to_num(upper, nan=1)

        self.add_constraint(lambda w: w >= self._lower_bounds)
        self.add_constraint(lambda w: w <= self._upper_bounds)

    def is_parameter_defined(self, parameter_name: str) -> bool:
        """
        Check if a parameter is defined
        """
        is_defined = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )

        for expr in objective_and_constraints:
            params = [arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)]
            for param in params:
                if param.name() == parameter_name and not is_defined:
                    is_defined = True
                elif param.name() == parameter_name and is_defined:
                    raise exceptions.InstantiationError(
                        "Parameter name defined multiple times"
                    )
        return is_defined

    
    def _get_all_args(expression: cp.Expression) -> List[cp.Expression]:
        """
        Helper function to recursively get all arguments from a cvxpy expression.
        """

        if expression.args == []:
            return [expression]
        else:
            return list(_flatten([_get_all_args(arg) for arg in expression.args]))
        
    def _flatten(alist: Iterable) -> Iterable:
        for item in alist:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                yield from _flatten(item)
            else:
                yield item

    def update_parameter_value(self, parameter_name: str, new_value: float) -> None:
        """
        Update the value of a parameter.
        """
        if not self.is_parameter_defined(parameter_name):
            raise exceptions.InstantiationError(
                f"Parameter {parameter_name} was not defined"
            )
        was_updated = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )
        for expr in objective_and_constraints:
            params = [arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)]
            for param in params:
                if param.name() == parameter_name:
                    param.value = new_value
                    was_updated = True
        if not was_updated:
            raise exceptions.instantiationError(
                f"Parameter {parameter_name} was not updated"
            )
    
    def _solve_cvxpy_opt_problem(self):
        """
        Helper method to solve the cvxpy problem and check output, once objectives and constraints have been defined
        """
        try:
            if self._opt is None:
                self._opt = cp.Problem(cp.Minimize(self._objective), self._constraints)
                self._initial_objective = self._opt.objective
                self._initial_constraint = list(self._constraints)
            else:
                if not self._objective is not self._initial_objective:
                    raise exceptions.InstantiationError(
                        "The objective function was changed after the initial optimization"
                        "Please create a new instance instead"
                    )
                
                constr_ids = {const.id for const in self._constraints}
                if not self._constraints != self._initial_constraints:
                    raise exceptions.InstantiationError(
                        "The constraints were changed after the initial optimization"
                        "Please create a new instance instead"
                    )
            self._opt.solve(
                solver=self._solver, verbose=self._verbose, **self._solver_options
            )
        except (TypeError, cp.DCPError) as e:
            raise exceptions.OptimizationError from e
        
        if self._opt.status not in {"optimal", "optimal_inaccurate"}:
            raise exceptions.OptimizationError(
                "Solver status: {}".format(self._opt.status)
            )
        self.weights = self._w.value.round(16) + 0.0
        return self._make_output_weights()
    
    def add_objective(self, new_objective: cp.Expression, **kwargs):
        """
        Add a new term into the objective function. This must be convex, and built from cvxpy atomic functions.
        """
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "It's not recommended to add objectives to an already solved problem which might have unintended consequences"
                "A new instance should be created for the new set of objectives."
            )
        self._additional_objectives.append(new_objective(self._w, **kwargs))

    def add_constraint(self, new_constraint: Callable[[cp.Variable], cp.Expression]):
        """
        Add a new constraint to the optimization problem. This must satisfy DCP rules.
        """
        if not callable(new_constraint):
            raise TypeError("New constraint must be a function (callable)")
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding constraints to an already solved problem is not supported"
                "A new instance should be created for the new set of constraints."
            )
        self._constraints.append(new_constraint(self._w))
    
    def convex_objective(self, custom_objective, weights_sum_to_one=True, **kwargs):
        """
        Optimize a custom convex objective function. Constraints should be added with 'ef.add_constraint()'. Optimizer arguments must be passed as keyword-args.
        """

        #custom_objective must have the right signature(w, **kwargs)
        self._objective = custom_objective(self._w, **kwargs)
        
        for obj in self._additional_objectives:
            self._objective += obj
        
        if weights_sum_to_one:
            self.add_constraint(lambda w: cp.sum(w) == 1)
        
        return self._solve_cvxpy_opt_problem()
    
def portfolio_performance(
    weights, expected_returns, cov_matrix, verbose=False, risk_free_rate=0.0
):
    """
    After optimising, calculate (and optionally print) the performance of the optimal
    portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

    :param expected_returns: expected returns for each asset. Can be None if
                            optimising for volatility only (but not recommended).
    :type expected_returns: np.ndarray or pd.Series
    :param cov_matrix: covariance of returns for each asset
    :type cov_matrix: np.array or pd.DataFrame
    :param weights: weights or assets
    :type weights: list, np.array or dict, optional
    :param verbose: whether performance should be printed, defaults to False
    :type verbose: bool, optional
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.0
    :type risk_free_rate: float, optional
    :raises ValueError: if weights have not been calculated yet
    :return: expected return, volatility, Sharpe ratio.
    :rtype: (float, float, float)
    """
    if isinstance(weights, dict):
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(range(len(expected_returns)))
        new_weights = np.zeros(len(tickers))

        for i, k in enumerate(tickers):
            if k in weights:
                new_weights[i] = weights[k]
        if new_weights.sum() == 0:
            raise ValueError("Weights add to zero, or ticker names don't match")
    elif weights is not None:
        new_weights = np.asarray(weights)
    else:
        raise ValueError("Weights is None")

    sigma = np.sqrt(objective_functions.portfolio_variance(new_weights, cov_matrix))

    if expected_returns is not None:
        mu = objective_functions.portfolio_return(
            new_weights, expected_returns, negative=False
        )

        sharpe = objective_functions.sharpe_ratio(
            new_weights,
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            negative=False,
        )
        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual volatility: {:.1f}%".format(100 * sigma))
            print("Sharpe Ratio: {:.2f}".format(sharpe))
        return mu, sigma, sharpe
    else:
        if verbose:
            print("Annual volatility: {:.1f}%".format(100 * sigma))
        return None, sigma, None