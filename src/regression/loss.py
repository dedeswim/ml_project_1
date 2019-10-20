# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""

import numpy as np
import math

def compute_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray, cf: str = "mse") -> float:
    """
    Calculate the loss using either MSE, RMSE or MAE for linear regression.

    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    w: ndarray
        Array containing the regression parameters to test.
    
    cf: str
        String indicating which cost function to use; "mse" (default), "rmse" or "mae".

    Returns
    -------
    loss: float
        The loss for the given regression parameters.
    """

    # Check whether the mode parameter is valid
    valid = ["mse", "rmse", "mae"]
    assert cf in valid, "Argument 'cf' must be either " + ", ".join(f"'{x}'" for x in valid)
    
    # Create the error vector (i.e. yn - the predicted n-th value)
    e = y - tx.dot(w)

    if "mse" in cf:
        mse = e.T.dot(e) / (2 * len(e))
        if cf == "rmse":
            return math.sqrt(2 * mse)
        return mse
    # mae
    return np.mean(np.abs(e))
