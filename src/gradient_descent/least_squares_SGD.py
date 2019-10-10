# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from helpers import batch_iter
from costs import compute_loss, compute_loss_mae
from gradient import compute_subgradient, compute_gradient
import numpy as np


def stochastic_gradient_descent(
        y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, batch_size: int, max_iters: int, ratio: float = 0.7) -> tuple:
    """Stochastic gradient descent algorithm. Uses MSE loss function.

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
        The ratio at wich the stepsize converges (0.5 - 1.0), default = 0.7.

    Returns
    -------
    losses, ws: ndarray, ndarray
        Array containing the losses using the different ws found with the SGD,
        Array containing the regression parameters found with the SGD.

    """

    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        
        # Calculate gamma (Robbins-Monroe condition)
        gamma = 1 / pow(n_iter + 1, ratio)

        # Create the batch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        print("Stochastic Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w


def stochastic_subgradient_descent(
        y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, batch_size: int, max_iters: int, ratio: float = 0.7) -> tuple:
    """Stochastic subgradient descent algorithm. Uses MAE loss function.
    
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    initial_w: ndarray
        Array containing the regression parameters to start from.
    batch_size: int
        Other size of the batches.
    max_iters: int
        The maximum number of iterations to be done.
    ratio: float
        The ratio at wich the stepsize converges (0.5 - 1.0), default = 0.7.

    Returns
    -------
    losses, ws: ndarray, ndarray:
        Array containing the losses using the different ws found with the SGD,
        Array containing the regression parameters found with the SGD.
    """

    # Define parameters to store w and loss
    ws = initial_w
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):

        # Calculate gamma (Robbins-Monroe condition)
        gamma = 1 / pow(n_iter + 1, ratio)

        # Create the batch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute gradient and loss
            subgradient = compute_subgradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss_mae(minibatch_y, minibatch_tx, w)

        # Update w by gradient
        w = w - gamma * subgradient

        print("Stochastic Subgradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w
