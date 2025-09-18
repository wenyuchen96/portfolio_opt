import warnings
import numpy as np
import pandas as pd

from .expected_returns import returns_from_prices

# Helper functions
def _is_positive_semidefinite(matrix: np.ndarray):
    """
    Check if a matrix is positive semidefinite("PSD").
    matrix: covariance to be tested
    """
    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False
    
def _pair_exp_cov(X: pd.Series, Y: pd.Series, span: int =180):
    """
    Calculate the exponential covariance between two time series of asset returns.
    X, Y: time series of asset returns
    span: the span of the exponential weighting function, default to 180
    """

    covariation =  (X - X.mean()) * (Y - Y.mean())
    #exponentially weight the covariation and take the mean
    if span < 10:
        warnings.warn("it's recommended to use a higher span, e.g. 30 days")
    return covariation.ewm(span=span).mean().iloc[-1]


def fix_nonpositive_semidefinite(matrix: np.ndarray, fix_method: str = "spectral"):
    """
    Check if a covariance matrix is positive semidefinite.
    If PSD, pass; if not, fix it with the selected method. 
    matrix: covariance matrix to be checked
    fix_method: {"spectral", "diag"}, default to "spectral"
    """
    if _is_positive_semidefinite(matrix):
        return matrix
    
    warnings.warn(
        "The covariance matrix is non-positive semidefinite. Amending eigenvalues."
    )

    #eigen-decomposition
    q, V = np.linalg.eigh(matrix)

    if fix_method == "spectral":
        #remove negative eigenvalues
        q = np.where(q > 0, q, 0)
        #reconstruct matrix
        fixed_matrix = V @ np.diag(q) @ V.T
    elif fix_method == "diag":
        min_eig = np.min(q)
        fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
    else:
        raise NotImplementedError(f"Method {fix_method} not implemented")
    
    if not _is_positive_semidefinite(fixed_matrix):
        warnings.warn("Could not fix the matrix. Please try a different risk model.", UserWarning)
    
    #rebuild labels if provided in the first place
    if isinstance(matrix, pd.DataFrame):
        return pd.DataFrame(fixed_matrix, index=matrix.index, columns=matrix.columns)
    else:
        return fixed_matrix

def cov_to_corr(cov_matrix: pd.DataFrame):
    """
    Convert a covariance matrix to a correlation matrix.
    
    cov_matrix: covariance matrix
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn("cov_matrix is not in a dataframe", RuntimeWarning)
        cov_matrix = pd.DataFrame(cov_matrix)
    
    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    corr = np.dot(Dinv, np.dot(cov_matrix, Dinv))
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.index)

def corr_to_cov(corr_matrix: pd.DataFrame, stdevs: np.ndarray):
    """
    Convert a correlation matrix to a covariance matrix.
    
    corr_matrix: correlation matrix
    stdevs: standard deviations of the assets
    """
    if not isinstance(corr_matrix, pd.DataFrame):
        warnings.warn("corr_matrix is not in a dataframe", RuntimeWarning)
        corr_matrix = pd.DataFrame(corr_matrix)

    return corr_matrix * np.outers(stdevs, stdevs)

# Main functions
def risk_matrix(prices: pd.DataFrame, method="sample_cov", **kwargs):
    """
    Calculate the covariance matrix based on the selected risk model specified in the method.

    prices: adjusted daily closing prices of the asset, each row represent the date and each column represents the ticker name
    method: risk model to be used. Including:
        - 'sample_cov'
        - 'semicovariance'
        - 'exp_cov'
        - 'ledoit_wolf'
        - 'ledoit_wolf_constant_variance'
        - 'ledoit_wolf_single_factor'
        - 'ledoit_wolf_constant_correlation'
        - 'oracle_approximating'
    """

    if method == "sample_cov":
        return sample_cov(prices, **kwargs)
    elif method == "semicovariance" or method == "semivariance":
        return semicovariance(prices, **kwargs)
    elif method == "exp_cov":
        return exp_cov(prices, **kwargs)
    # elif method == "ledoit_wolf" or method == "ledoit_wolf_constant_variance":
    #     pass
    # elif method == "ledoit_wolf_single_factor":
    #     pass
    # elif method == "ledoit_wolf_constant_correlation":
    #     pass
    # elif method == "oracle_approximating":
    #     pass
    else:
        raise NotImplementedError("Risk model {} not implemented".format(method))
    
def sample_cov(prices: pd.DataFrame, return_data=False, frequency=252, log_returns=False, **kwargs):
    """
    Calculate the annualized sample covariance matrix from daily asset returns.
    prices: adjusted daily closing prices of the asset, each row is a date and each column is the ticker name
    return_data: if true, the prices input is return instead of asset price.
    frequency: default to 252 trading days per year.
    log_returns: whether to calculate log returns; default to False
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    
    if return_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    return fix_nonpositive_semidefinite(
        returns.cov() * frequency, kwargs.get("fix_method", "spectral")
    )

def semicovariance(
    prices: pd.DataFrame,
    returns_data: bool = False,
    benchmark: float = 0.000079,
    frequency: int = 252,
    log_returns: bool = False,
    **kwargs
):
    """
    Estimate the semicovariance matrix, the covariance given that the returns are less than the benchmark.
    semicov = E([min(r_i - benchmark_return, 0)] . [min(r_j - benchmark_return, 0)])

    prices: adjusted daily closing prices of the asset, each row represent the date and each column represents the ticker
    return_data: if true, the prices input is return instead of asset price.
    benchmark: the benchmark return, default=daily risk-free rate; '1.02^(1/252) - 1'
    frequency: default to 252 trading days annually
    log_returns: whether to calculate log returns; default to False
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    drops = np.fmin(returns - benchmark, 0)
    T = drops.shape[0]
    return fix_nonpositive_semidefinite(
        (drops.T @ drops) / T * frequency, kwargs.get("fix_method", "spectral")
    )

def exp_cov(
    prices: pd.DataFrame,
    returns_data: bool = False,
    span: int = 180,
    frequency: int = 252,
    log_returns: bool = False,
    **kwargs      
):
    """
    Calculate the exponential covariance matrix from daily asset returns, with greater weight given to more recent data.
    
    prices: adjusted closing price of the asset, row is date and column is ticker name
    returns_data: if true, the prices input is return instead of asset price.
    span: the time span for the exponential moving average, default to 180 trading days.
    frequency: default to 252 trading days annually
    log_returns: whether to calculate log returns; default to False
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    assets = prices.columns
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    
    N = len(assets)
    #loop over matrix, filling entries with the pairwise exp cov
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = _pair_exp_cov(returns.iloc[:, i], returns.iloc[:, j], span)
    cov = pd.DataFrame(S*frequency, index=assets, columns=assets)

    return fix_nonpositive_semidefinite(cov, kwargs.get("fix_method", "spectral"))