"""
The 'plotting' module holds all the functions to generate plots. Including:

- 'plot_covariance': plot a covariance matrix
- 'plot_dendrogram': plot the hierarchical clusters in a portfolio
- 'plot_efficient_frontier': plot the efficient frontier
- 'plot_weights': bar chart of portfolio weights
"""

import warnings
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

from . import risk_models

# Helper functions
def _get_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except (ModuleNotFoundError, ImportError):
        raise ImportError("Please install plotly via pip to use interactive plots")
    
def _plot_io(**kwargs):
    """
    Save the plot to file.

    :param filename: name of the file to save to, defaults to None (no saving)
    :type filename: str, optional
    :param dpi: dpi of figure to save or plot, defaults to 300
    :type dpi: int (between 50-500)
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    """
    filename = kwargs.get("filename", None)
    dpi = kwargs.get("dpi", 300)
    showfig = kwargs.get("showfig", False)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(fname=filename, dpi=dpi)
    if showfig:
        plt.show()

# Core functions
def plot_covariance(
    cov_matrix: np.ndarray | pd.DataFrame, 
    plot_correlation=False,
    show_tickers=True,
    **kwargs
    ):
    """
    Plot a covariance matrix.

    cov_matrix: covariance matrix
    plot_correlation: whether to plot the correlation instead, default=False
    show_tickers: whether to show the tickers on the plot, default=True
    """

    if plot_correlation:
        matrix = risk_models.cov_to_corr(cov_matrix)
    else:
        matrix = cov_matrix
    fig, ax = plt.subplots()

    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)
        plt.xticks(rotation=90)

    _plot_io(**kwargs)

    return ax