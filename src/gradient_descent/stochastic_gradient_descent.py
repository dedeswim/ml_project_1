# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from helpers import batch_iter
from costs import compute_loss, compute_loss_mae
from gradient import compute_subgradient, compute_gradient


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm. Uses MSE loss function.

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
        The stepsize of the SGD

    Returns
    -------
    losses, ws: ndarray, ndarray
        Array containing the losses using the different ws found with the SGD,
        Array containing the regression parameters found with the SGD.
    
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):

        # Create the batch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return losses, ws


def stochastic_subgradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic subgradient descent algorithm. Uses MAE loss function.
    
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
        The stepsize of the SGD

    Returns
    -------
    losses, ws: ndarray, ndarray:
        Array containing the losses using the different ws found with the SGD,
        Array containing the regression parameters found with the SGD.
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):

        # Create the batch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute gradient and loss
            subgradient = compute_subgradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss_mae(minibatch_y, minibatch_tx, w)

        # Update w by gradient
        w = w - gamma * subgradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic Subgradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return losses, ws
