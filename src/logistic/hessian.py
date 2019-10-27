import numpy as np
from src.logistic.sigmoid import sigmoid


def compute_hessian(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float = 0) -> np.ndarray:
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
    hessian: np.ndarray
        The hessian for the given logistic linear parameters.
    """

    # Compute the sigmoid of the x.w vector
    sigmoid_txw = sigmoid(tx.dot(w)).reshape((tx.shape[0], 1))

    # Find the S diagonal matrix
    s = np.diag(np.diag(sigmoid_txw.dot((1 - sigmoid_txw).T)))

    # Compute regularizer component (if lambda != 0)
    regularizer = lambda_ * np.eye(w.shape[0]) if lambda_ else 0

    return tx.T.dot(s).dot(tx) + regularizer
