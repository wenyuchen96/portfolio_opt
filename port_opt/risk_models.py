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

def fix_nonpositive_semidefinite(matrix: np.ndarray, fix_method: str = "spectral"):
    """
    Check if a covariance matrix is positive semidefinite.
    If not, fix it with the selected method. 
    matrix: covariance matrix to be checked
    fix_method: {"spectral", "diag"}, default to "spectral"
    """
    if _is_positive_semidefinite(matrix):
        return matrix
    
    warnings.warn(
        "The covariance matrix is non-positive semidefinite. Amending eigenvalues."
    )

    #eigen-decomposition