# Portfolio Optimization

A Python toolkit for constructing and analysing Markowitz-style portfolios. The package bundles
utility functions for fetching market data, estimating expected returns and risk, optimising asset
weights, and visualising efficient frontiers.

## Highlights

- Return estimators (`mean_historical_return`, exponential moving averages, CAPM-based)
- Risk models (sample covariance, semicovariance, exponentially weighted covariance)
- Convex optimisation via `EfficientFrontier` plus a Critical Line Algorithm implementation
- Plotting helpers for efficient frontiers, covariance matrices, dendrograms, and weights
- Example notebooks demonstrating end-to-end workflows

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install cvxpy numpy pandas scipy yfinance matplotlib plotly
```

The project is developed against Python 3.10+, and relies on `cvxpy` for convex optimisation. When
working from a clone, either run scripts from the repository root or export the path so the
`port_opt` package resolves correctly:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```
