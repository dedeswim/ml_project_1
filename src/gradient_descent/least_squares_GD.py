# -*- coding: utf-8 -*-
"""Gradient Descent"""

from costs import compute_loss, compute_loss_mae
from gradient import compute_subgradient, compute_gradient
import numpy as np


def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple:
    """Gradient descent algorithm. Uses MSE loss function.

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
    losses, ws: ndarray, ndarray
        Array containing the losses using the different ws found with the GD,
        Array containing the regression parameters found with the GD.
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


def subgradient_descent(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: int) -> tuple:
    """Subgradient descent algorithm. Uses MAE loss function.
    
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
    losses, ws: ndarray, ndarray:
        Array containing the losses using the different ws found with the GD,
        Array containing the regression parameters found with the GD.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = compute_subgradient(y, tx, w)
        loss = compute_loss_mae(y, tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w
