import numpy as np
from typing import Tuple

from src.logistic.loss import compute_loss
from src.logistic.gradient import compute_gradient


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
        loss, gradient, w = gradient_descent_step(y, tx, w, gamma, lambda_)
        # log info
        if iteration % 100 == 0:
            print("Current iteration={i}, loss={loss}".format(i=iteration, loss=loss))
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


def gradient_descent_step(y: np.ndarray, tx: np.ndarray, w: np.ndarray, gamma: float,
                          lambda_: float = 0) -> Tuple[float, np.ndarray, np.ndarray]:
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

    Returns
    -------
    w: np.ndarray
        The linear parameters.
    
    loss: float
        The loss given w as parameters.
    """
    # Get loss, gradient, hessian
    loss = compute_loss(y, tx, w, lambda_=lambda_)
    gradient = compute_gradient(y, tx, w, lambda_=lambda_)

    # Update w
    w = w - gamma * gradient

    return loss, gradient, w
