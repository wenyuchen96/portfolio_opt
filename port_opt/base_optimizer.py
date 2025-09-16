import collections
import copy
import json
import warnings
from collections.abc import Iterable
from typing import List

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.optimize as sco

from .. import exceptions, objective_functions

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
        returns a custome deep copy of the optimizer
        """
        self_copy = copy.copy(self)