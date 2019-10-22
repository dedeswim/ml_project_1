from typing import Tuple

import numpy as np

from src.regression.loss import compute_loss
from src.regression.gradient import compute_gradient


def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, 
        max_iters: int, gamma: float) -> Tuple[float, np.ndarray]:
    """
    Gradient descent algorithm. Uses MSE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    initial_w: ndarray
        Array containing the regression parameters to start from.
    
    max_iters: int
        The maximum number of iterations to be done.
    
    gamma: float
        The stepsize of the GD

    Returns
    -------
    w: np.ndarray
        The regression parameters.
    
    loss: float
        The loss given w as parameters.
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w

def least_squares_SGD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
        batch_size: int, max_iters: int, ratio: float = 0.7) -> Tuple[float, np.ndarray]:
    """
    Stochastic gradient descent algorithm. Uses MSE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    initial_w: ndarray
        Array containing the regression parameters to start from.
    
    batch_size: int
        The size of the batches to be created.
    
    max_iters: int
        The maximum number of iterations to be done.
    
    ratio: float
        The ratio at which the stepsize converges (0.5 - 1.0), default = 0.7.

    Returns
    -------
    w: np.ndarray
        The regression parameters.
    
    loss: float
        The loss given w as parameters.
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):

        # Calculate gamma (Robbins-Monroe condition)
        gamma = 1 / pow(n_iter + 1, ratio)

        # TODO: get just one random row of tx
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        print("Stochastic Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w

def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the least squares solution.
    
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    Returns
    -------
    w: np.ndarray
        The regression parameters.
    
    loss: float
        The loss given w as parameters.
    """

    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)

    return loss, w


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> Tuple[float, np.ndarray]:
    """
    Computes ridge regression with the given `lambda_`.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    lambda_: float
        Lambda regularization parameter


    Returns
    -------
    w: np.ndarray
        The regression parameters.
    
    loss: float
        The loss given w as parameters.
    """

    lambda_p = lambda_ * 2 * tx.shape[0]

    w = np.linalg.solve(tx.T.dot(tx) + lambda_p * np.eye(tx.shape[1]), tx.T.dot(y))
    loss = compute_loss(y, tx, w)

    return loss, w
