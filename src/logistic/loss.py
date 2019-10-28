import numpy as np


def compute_logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float = 0) -> float:
    """"
    Calculates the loss for logistic linear.

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
    loss: float
        The loss for the given logistic linear parameters.
    """

    # Find the regularizer (if lambda != 0)
    regularizer = lambda_ / 2 * (np.linalg.norm(tx) ** 2) if lambda_ else 0

    summing = np.sum(np.log(1 + np.exp(tx.dot(w))))
    y_component = y.T.dot(tx.dot(w)).flatten().flatten()

    return summing - y_component + regularizer
