# -*- coding: utf-8 -*-
"""Some helper functions"""

import csv
import numpy as np
from typing import Tuple, List

from src.polynomials import build_poly_matrix_quadratic


def load_csv_data(data_path: str, sub_sample: bool = False, sub_sample_size=1000) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids).

    Parameters
    ----------
    data_path: str
        The path of the file containing the dataset.

    sub_sample: bool
        Whether the function should return the entire dataset (default behavior)
        or only the first 50 data points.

    sub_sample_size: int
        The size of the subsample.

    Returns
    -------
    y: ndarray
        Array that contains the correct values to be predicted.

    input_data: ndarray
        Matrix that contains the data points.

    ids: ndarray
        Array containing the id of the data points.
    """
    y = np.genfromtxt(data_path, delimiter=",",
                      skip_header=1, dtype=str, usecols=[1])
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones((len(y), 1))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::sub_sample_size]
        input_data = input_data[::sub_sample_size]
        ids = ids[::sub_sample_size]

    return yb, input_data, ids


def standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardizes the original data set.

    Parameters
    ----------
    x: ndarray
        Matrix that contains the data points to be standardized.

    Returns
    -------
    x: np.ndarray
        The standardized dataset

    mean_x: np.ndarray
        The mean of x before the standardization

    mean_x: np.ndarray
        The standard deviation of x before the standardization
    """
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def predict_labels(weights: np.ndarray, data: np.ndarray, mode: str = "logistic") -> np.ndarray:
    """
    Generates class predictions given weights, and a test data matrix.

    Arguments
    ---------
    weights: np.ndarray
        The weights of the predictive functions.

    data: np.ndarray
        The data for which the label must be predicted.

    mode: str
        The type of model, either 'logistic' (default) or 'linear'

    Returns
    -------
    y_pred: np.ndarray
        The predicted labels.
    """
    assert mode == "logistic" or "linear", "The model should be either logistic or linear"
    bound = 0 if mode == "linear" else 0.5
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= bound)] = -1 if mode == "linear" else 0
    y_pred[np.where(y_pred > bound)] = 1

    return y_pred


def create_csv_submission(ids: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """
    Creates an output file in csv format for submission to Kaggle.

    Arguments
    ---------
    ids: np.ndarray
        Event ids associated with each prediction.

    y_pred: np.ndarray
        Predicted class labels.

    name: np.ndarray
        String name of .csv output file to be created.
    """

    y_pred[np.where(y_pred == 0)] = -1

    with open(name, 'w') as csv_file:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csv_file, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def remove_incomplete_columns(x: np.ndarray, threshold: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes the columns that have more than threshold% missing values.

    Arguments
    ---------
    x: np.ndarray
        The matrix to be cleaned.

    threshold: float
        The maximum percentage of missing data allowed. The default one is 30%.

    Returns
    -------
    clean_x: np.ndarray
        The cleaned matrix.

    to_keep: nd.array
        Boolean array of the kept columns.
    """

    to_keep = np.apply_along_axis(
        count_missing_values, 0, x, [threshold]).flatten()

    return x[:, to_keep], to_keep


def count_missing_values(column: np.ndarray, threshold: float) -> bool:
    """
    Checks whether the percentage of missing values on a column is less than a given threshold.

    Parameters
    ----------
    column: np.ndarray
        The columns to be checked.

    threshold: float
        The maximum percentage of missing data.

    Returns
    -------
    valid: bool
        Whether the percentage of missing values on a column is less than threshold.
    """

    return (column == -999.0).sum() / column.shape[0] < threshold


def remove_correlated_columns(x: np.ndarray, threshold: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes correlated columns, leaving only one column for each originally correlated sets of columns.

    Arguments
    ---------
    x: np.ndarray
        The matrix to be cleaned.

    threshold: float
        The maximum percentage of correlation allowed (default = 0.9).

    Returns
    -------
    clean_x: np.ndarray
        The cleaned matrix.

    to_keep: nd.array
        Boolean array of the kept columns.
    """

    assert 0 <= threshold <= 1
    _, to_remove = np.where(np.triu(np.corrcoef(x.T), 1) > threshold)
    to_remove = set(to_remove)
    return np.delete(x, list(to_remove), axis=1), np.array([i not in to_remove for i in range(x.shape[1])])


def flatten_jet_features(x: np.ndarray, indexes: Tuple[int] = (4, 5, 6, 12, 23, 24, 25, 26, 27, 28)) -> np.ndarray:
    """
    Flattens the columns passed in indexes to one column, by mapping to 0 the entries with
    -999 and summing up the columns row-wise.

    Arguments
    ---------
    x: np.ndarray
        The matrix to be cleaned.

    indexes: float
        The indexes of the columns to be flattened (usually the jet number related ones)

    Returns
    -------
    x: np.ndarray
        The original array with the index columns flattened.
    """
    # Get the columns containing the jet number-related features
    jet_features_columns = x[:, indexes]

    for feature in indexes:
        jet_features_columns[np.where(x[:, feature] == -999)] = 0

    new_column = jet_features_columns.sum(axis=1).reshape((-1, 1))

    x_new_column = np.append(x, new_column, axis=1)

    return np.delete(x_new_column, indexes, axis=1)


def get_jet_indexes(x: np.ndarray) -> List[np.ndarray]:
    """
    Gets the masks with the rows belonging to each of the jet number subsets.

    Arguments
    ---------
    x: np.ndarray
        The array to be indexed.

    Returns
    -------
    indexes: List[np.ndarray]
        A list of ndarrays (of booleans). Each ndarray contains the mask of the relative
        subset. `indexes[0]` contains has True values on the rows with jet number 0, `indexes[1]`
        has True values on the rows with jet number 1 and `indexes[2]` has True values on the
        rows with jet numbers 2 and 3.
    """

    return [
        x[:, 22] == 0,
        x[:, 22] == 1,
        np.bitwise_or(x[:, 22] == 2, x[:, 22] == 3)
    ]


"""
Contains the indexes of the rows to be removed for each subset, since they contain null values.
"""
jet_indexes = [
    [4, 5, 6, 12, 23, 24, 25, 26, 27, 28],  # indexes to be removed from rows with jet number 0
    [4, 5, 6, 12, 26, 27, 28],  # indexes to be removed from rows with jet number 1
    []  # no indexes are removed, since with jet numbers 2 and 3 no features are missing
]


def compute_accuracy(tx: np.ndarray, w: np.ndarray, y: np.ndarray, mode: str = "logistic") -> float:
    """
    Computes the accuracy on the given dataset using the given model parameters

    Arguments
    ---------
    tx: np.ndarray
        The data with which compute the accuracy.
    w: np.ndarray
        The weights of the model to predict with.
    y: np.ndarray
        The labels to be predicted.
    mode: np.ndarray
        The kind of model being used. It must be either `logistic` or `linear`.

    Returns
    -------
    accuracy: float
        The accuracy.
    """

    assert mode == "logistic" or "linear", "The model should be either logistic or linear"

    # Predict the labels
    y_pred = predict_labels(w, tx, mode=mode)

    # Get the number of elements for which the prediction is equal to the original label,
    # and divide it by the total number of elements.
    return (y_pred == y).sum() / y_pred.shape[0]


def clean_mass_feature(x: np.ndarray) -> np.ndarray:
    """
    Deals with the fact that some mass entries are null. It first creates a new column
    where each element is 0 if the corresponding row has a non-null (!=-999) value,
    otherwise it is 1. Then it substitutes the null values in the original mass column
    and adds the newly created column to the original dataset.

    Arguments
    ---------
    x: np.ndarray
        The dataset whose masses features need to be fixed.

    Returns
    -------
    x: np.ndarray
        The array with fixed mass values and with the new column.
    """
    # Create a new array containing all 0
    x_mass = np.zeros(x.shape[0])

    # Set the elements corresponding to rows with missing mass to 1
    x_mass[x[:, 0] == -999] = 1

    # Set the missing values to the median
    x[:, 0][x[:, 0] == -999] = np.median(x[:, 0][x[:, 0] != -999])

    # Add the newly created column to the dataset
    x = np.column_stack((x, x_mass))

    return x


def prepare_x(x: np.ndarray, indexes: List[np.ndarray], i: int):
    """
    Prepares the i-th subset of x to be used for training. In particular, it:

    * deletes the columns with missing values.
    * takes the logarithm of each feature.
    * standardizes the data.
    * builds the 2nd degree expansion of the rows.
    * adds a 1s column to be multiplied with a w_0 weight.

    Arguments
    ---------
    x: np.ndarray
        The dataset whose masses features need to be fixed.

    indexes: np.ndarray
        the mask containing the row indexes to be taken in consideration.

    i: int
        The subset to take in consideration.

    Returns
    -------
    x: np.ndarray
        The array ready to be used for training.
    """
    # Get the rows relative to the i-th subset taken in consideration
    tx_i = x[indexes[i]]

    # Delete the columns that are -999 for the given subset
    tx_del = np.delete(tx_i, jet_indexes[i], axis=1)

    # Take the logarithm of each column
    for li in range(tx_del.shape[1]):
        tx_del[:, li] = np.apply_along_axis(lambda n: np.log(
            1 + abs(tx_del[:, li].min()) + n), 0, tx_del[:, li])

    # Standardize the data
    tx_std = standardize(tx_del)[0]

    # Build the polynomial expansion of degree 2 and add the 1s column
    tx = build_poly_matrix_quadratic(tx_std)
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]

    return tx
