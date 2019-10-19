# -*- coding: utf-8 -*-
"""Some helper functions"""

import csv
from typing import Iterable, Tuple
import numpy as np

def load_csv_data(data_path: str, sub_sample: bool = False) \
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
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

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


def predict_labels(weights: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Generates class predictions given weights, and a test data matrix.

    Arguments
    ---------
    weights: np.ndarray
        The weights of the predictive functions.
    
    data: np.ndarry
        The data for which the label must be predicted.
    
    Returns
    -------
    y_pred: np.ndarray
        The predicted labels.    
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

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
