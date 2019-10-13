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
        The ratio between the number of wanted training data points and the total number of data points
    seed: float
    """
    # Set seed
    np.random.seed(seed)

    # Get the training set size
    train_size = round(x.shape[0] * ratio)
    # Get an array containing all the indexes of the arrays
    indexes = np.arange(x.shape[0])
    # Shuffle the indexes
    np.random.shuffle(indexes)

    # Get the number of wanted indexes from the shuffled ones
    train_indexes = indexes[:train_size]
    # Get the other indexes using set difference
    test_indexes = np.setdiff1d(indexes, train_indexes)

    # Create training set
    x_train = x[train_indexes]
    y_train = y[train_indexes]

    # Create test set
    x_test = x[test_indexes]
    y_test = y[test_indexes]

    return x_train, x_test, y_train, y_test
