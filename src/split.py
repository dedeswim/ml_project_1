# -*- coding: utf-8 -*-
"""Contains a function to split data sets between training and test sets"""

from typing import Tuple
import numpy as np


def split_data(x: np.ndarray, y: np.ndarray, ratio: float, seed: float = 1) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing

    Parameters
    ----------
    x : np.ndarray
        The array containing the features.

    y : np.ndarray
        The array containing the targets

    ratio: float
        The ratio between the number of wanted training data points and
        the total number of data points.

    seed: float
        The seed with which the random generator is initialized.

    Returns
    -------
    x_train, y_train, x_test, y_test : ndarray
    """
    # Set seed
    np.random.seed(seed)

    # Randomly choose indexes of train set
    data_len = x.shape[0]
    idxs = np.random.choice(data_len, size=round(
        data_len * ratio), replace=False)

    # Create a mask from indexes
    mask = np.zeros(data_len, dtype=bool)
    mask[idxs] = True

    #      x_train  y_train  x_test    y_test
    return x[mask], y[mask], x[~mask], y[~mask]
