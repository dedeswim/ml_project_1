import numpy as np
from typing import Tuple

from src.logistic.loss import compute_logistic_loss
from src.logistic.gradient import compute_logistic_gradient
from src.linear.loss import compute_loss
from src.linear.gradient import compute_gradient
from random import randrange


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float,
                            initial_w: np.ndarray, max_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    """
    Does the regularized logistic linear.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.

    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.

    initial_w: ndarray
        Array containing the linear parameters to start with.

    max_iters: int
        The maximum number of iterations to do.

    gamma: float
        Gradient descent stepsize

    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """

    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic linear
    for iteration in range(max_iters):
        # get loss and update w.
        loss, gradient, w = gradient_descent_step(y, tx, w, gamma, lambda_, mode='logistic')
        # log info
        if iteration % 100 == 0:
            print("Current iteration={i}, loss={loss}".format(
                i=iteration, loss=loss))
            print("||d|| = {d}".format(d=np.linalg.norm(gradient)))
        # converge criterion
        losses.append(loss)
        # print("Current iteration={i}, loss={l}, ||d|| = {d}".format(i=iter, l=loss, d=np.linalg.norm(gradient)))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    # visualization
    print("loss={l}".format(l=compute_loss(y, tx, w)))

    return w, losses[-1]


def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                        max_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    """
    Computes the parameters for the logistic linear.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.

    initial_w: ndarray
        Array containing the linear parameters to start with.

    max_iters: int
        The maximum number of iterations to do.

    gamma: float
        Gradient descent stepsize

    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """

    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                     max_iters: int, gamma: float) -> Tuple[float, np.ndarray]:
    """
    Gradient descent algorithm. Uses MSE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.

    initial_w: ndarray
        Array containing the linear parameters to start from.

    max_iters: int
        The maximum number of iterations to be done.

    gamma: float
        The stepsize of the GD

    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
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


def least_squares_SGD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                      max_iters: int, gamma: float) -> Tuple[float, np.ndarray]:
    """
    Stochastic gradient descent algorithm. Uses MSE loss function.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.

    initial_w: ndarray
        Array containing the linear parameters to start from.

    max_iters: int
        The maximum number of iterations to be done.

    gamma: float
        The stepsize of the SGD.

    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    y_len = len(y)
    for n_iter in range(max_iters):

        rand_idx = randrange(y_len)
        rand_tx = tx[rand_idx]
        rand_y = y[rand_idx]

        gradient = compute_gradient(rand_y, rand_tx, w)
        loss = compute_loss(y, tx, w)

        # Update w by gradient
        w = w - gamma * gradient

        print("Stochastic Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))

    return loss, w


def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the least squares solution.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """

    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)

    return loss, w


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> Tuple[float, np.ndarray]:
    """
    Computes ridge linear with the given `lambda_`.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.

    lambda_: float
        Lambda regularization parameter


    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """
    
    # Compute the lambda
    lambda_p = lambda_ * 2 * tx.shape[0]
    
    # Solve the linear system
    w = np.linalg.solve(tx.T.dot(tx) + lambda_p *
                        np.eye(tx.shape[1]), tx.T.dot(y))
    
    # Compute the loss
    loss = compute_loss(y, tx, w, lambda_)
    
    return loss, w



def gradient_descent_step(y: np.ndarray, tx: np.ndarray, w: np.ndarray, gamma: float,
                          lambda_: float = 0, mode='linear') -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes one step of gradient descent.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.

    w: ndarray
        Array containing the linear parameters to test.

    gamma: float
        The stepsize.

    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.
    
    mode: str
        The kind of gradient descent. Must be either `linear` or `logistic`

    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """
    # Get loss, gradient, hessian
    loss = (compute_loss(y, tx, w, lambda_=lambda_) if mode == 'linear'
            else compute_logistic_loss(y, tx, w, lambda_=lambda_))
    
    gradient = (compute_gradient(y, tx, w) if mode == 'linear'
               else compute_logistic_gradient(y, tx, w, lambda_=lambda_))

    # Update w
    w = w - gamma * gradient

    return loss, gradient, w
