# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np


def compute_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """Calculate the loss using MSE for linear regression.

    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    w: ndarray
        Array containing the regression parameters to test.

    Returns
    -------
    loss: float
        The loss for the given regression parameters.
    """
    e = y - tx.dot(w)
    n = y.shape[0]

    return 1 / (2 * n) * e.T.dot(e)


def compute_loss_mae(y, tx, w):
    """Calculate the loss using MAE for linear regression.

    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    w: ndarray
        Array containing the regression parameters to test.

    Returns
    -------
    loss: float
        The loss for the given regression parameters.
    """

    e = y - tx.dot(w)
    n = y.shape[0]

    return 1 / n * np.abs(e).sum()
