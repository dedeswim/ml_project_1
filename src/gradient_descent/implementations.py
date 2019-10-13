from costs import compute_loss
from gradient import compute_gradient, compute_subgradient
from helpers import batch_iter
import numpy as np

def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple:
    """Gradient descent algorithm. Uses MSE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    initial_w: ndarray
        Array containing the regression parameters to start from.
    max_iters: int
        The maximum number of iterations to be done.
    gamma: float
        The stepsize of the GD

    Returns
    -------
    losses, ws: ndarray, ndarray
        Array containing the losses using the different ws found with the GD,
        Array containing the regression parameters found with the GD.
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w


def subgradient_descent(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: int) -> tuple:
    """Subgradient descent algorithm. Uses MAE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    initial_w: ndarray
        Array containing the regression parameters to start from.
    max_iters: int
        The maximum number of iterations to be done.
    gamma: float
        The stepsize of the GD

    Returns
    -------
    losses, ws: ndarray, ndarray:
        Array containing the losses using the different ws found with the GD,
        Array containing the regression parameters found with the GD.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = compute_subgradient(y, tx, w)
        loss = compute_loss(y, tx, w, "mae")

        # Update w by gradient
        w = w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w


def stochastic_gradient_descent(
        y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, batch_size: int, max_iters: int, ratio: float = 0.7) -> tuple:
    """Stochastic gradient descent algorithm. Uses MSE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    initial_w: ndarray
        Array containing the regression parameters to start from.
    batch_size: int
        The size of the batches to be created.
    max_iters: int
        The maximum number of iterations to be done.
    ratio: float
        The ratio at wich the stepsize converges (0.5 - 1.0), default = 0.7.

    Returns
    -------
    losses, ws: ndarray, ndarray
        Array containing the losses using the different ws found with the SGD,
        Array containing the regression parameters found with the SGD.

    """

    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):

        # Calculate gamma (Robbins-Monroe condition)
        gamma = 1 / pow(n_iter + 1, ratio)

        # Create the batch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        print("Stochastic Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w


def stochastic_subgradient_descent(
        y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, batch_size: int, max_iters: int, ratio: float = 0.7) -> tuple:
    """Stochastic subgradient descent algorithm. Uses MAE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    initial_w: ndarray
        Array containing the regression parameters to start from.
    batch_size: int
        Other size of the batches.
    max_iters: int
        The maximum number of iterations to be done.
    ratio: float
        The ratio at wich the stepsize converges (0.5 - 1.0), default = 0.7.

    Returns
    -------
    losses, ws: ndarray, ndarray:
        Array containing the losses using the different ws found with the SGD,
        Array containing the regression parameters found with the SGD.
    """

    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):

        # Calculate gamma (Robbins-Monroe condition)
        gamma = 1 / pow(n_iter + 1, ratio)

        # Create the batch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute gradient and loss
            subgradient = compute_subgradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w, cf='mae')

        # Update w by gradient
        w = w - gamma * subgradient

        print("Stochastic Subgradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w

def least_squares(y, tx):
    """calculate the least squares solution."""

    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = compute_loss(y, tx, w, "mse")

    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    lambda_p = lambda_ * 2 * tx.shape[0]

    w_ridge = np.linalg.solve(tx.T.dot(tx) + lambda_p * np.eye(tx.shape[1]), tx.T.dot(y))
    mse = compute_loss(y, tx, w_ridge)

    return mse, w_ridge
