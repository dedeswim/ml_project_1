# -*- coding: utf-8 -*-
"""Some helper functions"""

import csv
from typing import Iterable, Tuple, List
import numpy as np


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
        or only the first 50 datapoints.

    Returns
    -------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    input_data: ndarray
        Matrix that contains the data points.
    
    ids: ndarray
        Array containing the id of the datapoints.
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=[1])
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
    x: np.ndarry
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
    
    data: np.ndarry
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
    y_pred[np.where(y_pred <= bound)] = -1
    y_pred[np.where(y_pred > bound)] = 1

    return y_pred


def create_csv_submission(ids: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """
    Creates an output file in csv format for submission to Kaggle.
    
    Arguments
    ---------
    ids: np.ndarray
        Event ids associated with each prediction.
    
    y_pred: np.ndarry
        Predicted class labels.
    
    name: np.ndarray
        String name of .csv output file to be created.
    """
    with open(name, 'w') as csv_file:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csv_file, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def remove_incomplete_columns(x: np.ndarray, threshold: float = 0.30) -> np.ndarray:
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
    
    to_keep = np.apply_along_axis(count_missing_values, 0, x, [threshold]).flatten()
        
    return x[:, to_keep], to_keep
        
        
def count_missing_values(column: np.ndarray, threshold: float) -> bool:
    """
    Checks whether the percentage of missing values on a column is less than a given threshold.
    
    Parameters
    ----------
    x: np.ndarray
        The columns to be checked.
    
    threshold: float
        The maximum percentage of missing data.
    
    Returns
    -------
    valid: bool
        Whether the percentage of missing values on a column is less than threshold.
    """

    return (column == -999.0).sum() / column.shape[0] < threshold


def remove_correlated_columns(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
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
    to_remove, _ = np.where(np.triu(np.corrcoef(x.T), 1) > threshold)
    to_remove = set(to_remove)
    return np.delete(x, list(to_remove), axis=1), np.array([i in to_remove for i in range(x.shape[1])])