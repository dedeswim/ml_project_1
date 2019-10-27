import numpy as np
from typing import Callable, Tuple

from src.linear.loss import compute_loss
from src.helpers import compute_accuracy


def build_k_indices(y: np.ndarray, k_fold: int, seed: int) -> np.ndarray:
    """
    Builds k indices for k-fold cross validation.

    Arguments
    ---------
    y: np.ndarray
        The labels vector (used to get the size of the dataset)

    k_fold: int
        The number of folds

    seed: int
        The seed used to initialize the pseudo-random numbers generator
    Returns
    -------
    k_indices: np.ndarray
        The array containing the indices of each fold.
    """
    # Take the number of rows
    num_row = y.shape[0]

    # Get the interval of each fold
    interval = int(num_row / k_fold)
    np.random.seed(seed)

    # Get a permutation of the indices
    indices = np.random.permutation(num_row)

    # Create an array of indexes from the permutated ones
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


def cross_validation(y: np.ndarray, x: np.ndarray, k_indices: np.ndarray, k: int, lambda_: float,
                     model: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds k indices for k-fold cross validation.

    Arguments
    ---------
    x: np.ndarray
        The dataset on which the training must be performed.

    y: np.ndarray
        The target labels.
    k: int
        The number of folds.

    k_indices: np.ndarray
        the indices to be used for the folds of cross validation.

    lambda_: float
        The regularizer's lambda.

    model: Callable
        The function to be used for the training.
    Returns
    -------
    k_indices: np.ndarray
        The array containing the indices of each fold.
    """

    losses_tr, losses_te, accs_tr, accs_te, ws = [], [], [], [], []

    for k_ in range(k):
        test_indices = k_indices[k_]
        train_indices = np.setdiff1d(k_indices.flatten(), test_indices)

        y_train = y[train_indices]
        x_train = x[train_indices]
        y_test = y[test_indices]
        x_test = x[test_indices]

        # Ridge linear
        loss_tr, w = model(y_train, x_train, lambda_)

        # Calculate the loss for test data
        loss_te = compute_loss(y_test, x_test, w)

        # Compute the accuracy for both test and train set
        acc_tr = compute_accuracy(x_train, w, y_train, mode='linear')
        acc_te = compute_accuracy(x_test, w, y_test, mode='linear')

        # Save the accuracy of the current fold
        accs_te.append(acc_te)
        accs_tr.append(acc_tr)

        # Save the RMSE of the current fold
        losses_tr.append(np.math.sqrt(2 * loss_tr))
        losses_te.append(np.math.sqrt(2 * loss_te))

        # Save the w of the current fold
        ws.append(w)

    # Return the mean of the ws and the means of RMSEs and accuracies
    return np.mean(ws, axis=0), np.mean(losses_tr), np.mean(losses_te), np.mean(accs_tr), np.mean(accs_te)
