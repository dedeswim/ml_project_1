from typing import Tuple

import numpy as np
from random import randrange

from src.linear.loss import compute_loss
from src.linear.gradient import compute_subgradient


def subgradient_descent(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                        max_iters: int, gamma: int) -> Tuple[float, np.ndarray]:
    """
    Subgradient descent algorithm. Uses MAE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    initial_w: ndarray
        Array containing the linear parameters to start from.
    
    max_iters: int
        The maximum number of iterations to be done.
    
    gamma: float
        The stepsize of the GD

    Returns
    -------
    w: np.ndarray
        The linear parameters.
    
    loss: float
        The loss given w as parameters.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = compute_subgradient(y, tx, w)
        loss = compute_loss(y, tx, w, "mae")

        # Update w by gradient
        w = w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w


def stochastic_subgradient_descent(
        y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int,
        ratio: float = 0.7) -> Tuple[float, np.ndarray]:
    """
    Stochastic subgradient descent algorithm. Uses MAE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    initial_w: ndarray
        Array containing the linear parameters to start from.
    
    max_iters: int
        The maximum number of iterations to be done.
    
    ratio: float
        The ratio at which the stepsize converges (0.5 - 1.0), default = 0.7.

    Returns
    -------
    w: np.ndarray
        The linear parameters.
    
    loss: float
        The loss given w as parameters.
    """

    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        # Calculate gamma (Robbins-Monroe condition)
        gamma = 1 / pow(n_iter + 1, ratio)

        rand_idx = randrange(y.shape[0])
        rand_tx = tx[rand_idx]
        rand_y = y[rand_idx]

        # Compute gradient and loss
        subgradient = compute_subgradient(rand_y, rand_tx, w)
        loss = compute_loss(rand_y, rand_tx, w, cf='mae')

        # Update w by gradient
        w = w - gamma * subgradient

        print("Stochastic Subgradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w