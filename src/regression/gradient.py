import numpy as np


def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the MSE

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    w: ndarray
        Array containing the parameters of the linear model, from w0 on.

    Returns
    -------
    gradient: ndarray
        Array containing the gradient of the MSE function.
    """

    # Get the number of datapoints
    n = y.shape[0]
    
    # Create the error vector (i.e. yn - the predicted n-th value)
    e = y - tx.dot(w)

    return - 1 / n * tx.T.dot(e)


def compute_subgradient(y, tx, w):
    """
    Computes a subgradient for the MAE.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    w: ndarray
        Array containing the parameters of the linear model, from w0 on.

    Returns
    -------
    gradient: ndarray
        Array containing the gradient of the MAE function.
    """

    # Get the number of datapoints
    n = y.shape[0]
    
    # Create the error vector (i.e. yn - the predicted n-th value)
    e = y - tx.dot(w)

    return - 1 / n * tx.T.dot(np.sign(e))
