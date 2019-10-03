# -*- coding: utf-8 -*-
"""Gradient Descent"""

from costs import compute_loss, compute_loss_mae
from gradient import compute_subgradient, compute_gradient
from numpy import ndarray


def gradient_descent(y: ndarray, tx: ndarray, initial_w: ndarray, max_iters: int, gamma: float) -> ndarray:
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
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return losses, ws


def subgradient_descent(y: ndarray, tx: ndarray, initial_w: ndarray, max_iters: int, gamma: int):
    """Subgradient descent algorithm. It uses MSE loss function.
    
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
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = compute_subgradient(y, tx, w)
        loss = compute_loss_mae(y, tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return losses, ws
