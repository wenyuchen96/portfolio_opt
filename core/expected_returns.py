import warnings
import numpy as np
import pandas as pd

# Helper functions
def _check_returns(returns: pd.DataFrame):
    # Checking if NaNs in returns excluding starting NaNs
    if np.any(np.isnan(returns.mask(returns.ffill().isnull(), 0))):
        warnings.warn("Some returns are NaNs. Please check the price data.", UserWarning)
    if np.any(np.isinf(returns)):
        warnings.warn("Some returns are infinite. Please check the price data.", UserWarning)

def returns_from_prices(prices: pd.DataFrame, log_returns: bool = False):
    """
    Calculate arithmetic or log returns from prices. 

    prices: adjusted daily closing prices of the asset, each row represent the date and each column represents the ticker name
    log_returns: whether to calculate log returns; default to False
    """
    if log_returns:
        returns = np.log(1 + prices.pct_change(fill_method=None)).dropna(how='all')
    else:
        returns = prices.pct_change(fill_method=None).dropna(how='all')
    return returns

def prices_from_returns(returns: pd.DataFrame, log_returns: bool = False):
    """
    Calculate pseudo-prices from returns, which are not true asset prices because the initial prices are assumed as 1.

    returns: daily returns of the asset
    log_returns: whether to calculate log returns; default to False
    """
    if log_returns:
        ret = np.exp(returns)
    else:
        ret = (1 + returns)
    ret.iloc[0] = 1 # initial asset prices set to 1
    return ret.cumprod()

# Main functions
def return_model(prices: pd.DataFrame, method="mean_historical_return", **kwargs):
    """
    Compute the asset returns based on the selected model in the method.

    prices: adjusted daily closing prices of the asset, each row represent the date and each column represents the ticker name
    method: calculation method of the asset return. Including:
        - 'mean_historical_return'
        - 'ema_historical_return'
        - 'capm_return'
    """

    if method == "mean_historical_return":
        return mean_historical_return(prices, **kwargs)
    elif method == "ema_historical_return":
        return ema_historical_return(prices, **kwargs)
    elif method == "capm_return":
        return capm_return(prices, **kwargs)
    else:
        raise NotImplementedError("Return model {} not implemented".format(method))
    
def mean_historical_return(
    prices: pd.DataFrame, 
    returns_data: bool = False, 
    compounding: bool = False, 
    frequency: int = 252, 
    log_returns: bool = False
    ):
    """
    Calculate the annualized mean historical return from daily asset prices.

    prices: adjusted daily closing prices of the asset, each row represent the date and each column represents the ticker name
    returns_data: if true, the prices input is return instead of asset price.
    compounding: if true, output geometric mean returns; if false, output arithmetic mean returns.
    frequency: default to 252 trading days per year.
    log_returns: whether to calculate log returns; default to False
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    _check_returns(returns)
    if compounding:
        return (1 + returns).prod() ** (frequency / returns.count()) - 1
    else:
        return returns.mean() * frequency
    
def ema_historical_return(
    prices: pd.DataFrame,
    returns_data: bool = False,
    compounding: bool = True,
    span: int = 500,
    frequency: int = 252,
    log_returns: bool = False
    ):
    """
    Calculate the exponential weighted moving average historical return from daily asset prices.

    prices: adjusted daily closing prices of the asset, each row represent the date and each column represents the ticker name
    returns_data: if true, the prices input is return instead of asset price.
    compounding: if true, output geometric mean returns; if false, output arithmetic mean returns.
    span: the time span for the exponential moving average, default to 500 trading days.
    frequency: default to 252 trading days per year.
    log_returns: whether to calculate log returns; default to False
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    _check_returns(returns)
    if compounding:
        return (1 + returns.ewm(span=span).mean().iloc[-1]) ** frequency - 1
    else:
        return returns.ewm(span=span).mean().iloc[-1] * frequency
    
def capm_return(
    prices: pd.DataFrame,
    market_prices: pd.DataFrame = None,
    returns_data: bool = False,
    risk_free_rate: float = 0.0,
    compounding: bool = False,
    frequency: int = 252,    
    log_returns: bool = False
    ):
    """
    Compute the return based on the Capital Asset Pricing Model (CAPM).
    R_i = R_f + \\beta * (R_m - R_f)

    prices: adjusted daily closing prices of the asset, each row represent the date and each column represents the ticker name
    market_prices: adjusted closing prices of the market benchmark, default to None
    returns_data: if true, the prices input is return instead of asset price.
    risk_free_rate: default to 0.0; select the approriate time period corresponding to the frequency parameter.
    compounding: if true, output geometric mean returns; if false, output arithmetic mean returns.
    frequency: default to 252 trading days per year.
    log_returns: whether to calculate log returns; default to False
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    
    market_returns = None

    if returns_data:
        returns = prices.copy()
        if market_prices is not None:
            market_returns = market_prices
    else:
        returns = returns_from_prices(prices, log_returns)

        if market_prices is not None:
            if not isinstance(market_prices, pd.DataFrame):
                warnings.warn("market_prices are not in a dataframe", RuntimeWarning)
                market_prices = pd.DataFrame(market_prices)
            market_returns = returns_from_prices(market_prices, log_returns)
    
    # Use the equal weighted asset return as the proxy for market return
    if market_returns is None:
        returns["mkt"] = returns.mean(axis=1)
    else:
        market_returns.columns = ["mkt"]
        returns = returns.join(market_returns, how="left")

    _check_returns(returns)

    # Covariance matrix for individual asset and market
    cov = returns.cov()
    betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    betas = betas.drop("mkt")

    if compounding:
        mkt_mean_ret = (1 + returns["mkt"]).prod() ** (frequency / returns["mkt"].count()) - 1
    else:
        mkt_mean_ret = returns["mkt"].mean() * frequency
    # CAPM formula
    return risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)