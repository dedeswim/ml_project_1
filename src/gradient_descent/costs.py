# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using MSE.
    """
    e = y - tx.dot(w)
    n = y.shape[0]

    return 1 / (2 * n) * e.T.dot(e)


def compute_loss_mae(y, tx, w):
    """Calculate the loss using MAE.
    """

    e = y - tx.dot(w)
    n = y.shape[0]

    return 1 / n * np.abs(e).sum()
