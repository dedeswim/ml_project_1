# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""

import numpy as np


def compute_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray, cf: str = "mse") -> float:
    """
    Calculate the loss using either MSA or MAE for linear regression.

    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    w: ndarray
        Array containing the regression parameters to test.
    cf: str
        String indicating which cost function to use; "mse" (default) or "mae".

    Returns
    -------
    loss: float
        The loss for the given regression parameters.
    """

    assert cf == "mse" or cf == "mae", "Argument 'cf' must be either 'mse' or 'mae'"
    e = y - tx.dot(w)
    return e.T.dot(e) / (2 * len(e)) if cf == "mse" else np.mean(np.abs(e))
