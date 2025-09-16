"""
The objective_functions.py houses optimization objectives, which includes the actual objective functions called by the 'EfficientFrontier' object's optimization methods. Including:
    - Portfolio variance
    - Portfolio return
    - Sharpe ratio
    - L2 regularization
    - Quadratic utility
    - Transaction cost model
    - ex-ante (squared) tracking error
    - ex-post (squared) tracking error
"""

import cvxpy as cp
import numpy as np

def _objective_value(w: np.ndarray | cp.Variable, obj: cp.Expression):
    """
    Return either the value of the objective function or the objective function as a cvxpy object, depending on whether w is a cvxpy variable or np array.
    """
    
    if isinstance(w, np.ndarray):
        if np.isscalar(obj):
            return obj
        elif np.isscalar(obj.value):
            return obj.value
        else:
            return obj.value.item()
    else:
        return obj
    
def portfolio_variance(w: np.ndarray | cp.Variable, cov_matrix: np.ndarray):
    """
    Calculate the portfolio variance.

    w: asset weights in the portfolio
    cov_matrix: covariance matrix of the assets in the portfolio
    """
    variance = cp.quad_form(w, cov_matrix)
    return _objective_value(w, variance)

def portfolio_return(w: np.ndarray | cp.Variable, expected_returns: np.ndarray, negative: bool = True):
    """"
    Calculate the negative mean portfolio return, since minimizing the negative is equivalent to maximizing the positive.

    w: asset weights in the portfolio
    expected_returns: expected returns of the assets in the portfolio
    """

    sign = -1 if negative else 1
    mu = w @ expected_returns
    return _objective_value(w, sign * mu)

def sharpe_ratio(w: np.ndarray | cp.Variable, expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.0, negative: bool = True):
    """
    Calculate the Sharpe ratio.

    w: asset weights in the portfolio
    expected_returns: expected returns of the assets in the portfolio
    cov_matrix: covariance matrix of the assets in the portfolio
    risk_free_rate: default to 0.0; select the approriate time period corresponding to the frequency parameter.
    negative: whether quantity should be negative, so we can minimize the negative Sharpe ratio.
    """

    mu = w @ expected_returns
    sigma = cp.sqrt(cp.quad_form(w, cov_matrix, assume_PSD=True))
    sign = -1 if negative else 1
    return _objective_value(w, sign * (mu - risk_free_rate) / sigma)

def L2_reg(w: np.ndarray | cp.Variable, gamma=1):
    """
    L2 regularization of the portfolio, to increase the number of nonzero weights

    w: asset weights in the portfolio
    gamma: L2 regularization parameter, default to 1. Increase if want more non-negligible weights.
    """

    L2_reg = gamma * cp.sum_squares(w)
    return _objective_value(w, L2_reg)