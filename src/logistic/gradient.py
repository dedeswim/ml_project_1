import numpy as np
from src.logistic.sigmoid import sigmoid


def compute_logistic_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float = 0) -> np.ndarray:
    """"
    Calculates the of logistic linear loss.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.

    w: ndarray
        Array containing the linear parameters to test.

    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.

    Returns
    -------
    gradient: np.ndarray
        The gradient for the given logistic linear parameters.
    """

    # Find the regularizer component (if lambda != 0)
    regularizer = lambda_ * w if lambda_ else 0

    return tx.T.dot(sigmoid(tx.dot(w)) - y) + regularizer
