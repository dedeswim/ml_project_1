import numpy as np
from typing import Tuple
from random import randrange

from src.logistic.loss import compute_loss
from src.logistic.gradient import compute_gradient
from src.logistic.hessian import compute_hessian


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, lambda_: float,
                            max_iters: int, gamma: float, method: str = 'sgd', ratio: float = 0.7) -> Tuple[np.ndarray, float]:
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
        Gradient descent stepsize. Only used for gd.

    method: str
        The method for the optimization solution. Should be either SGD, Newton or GD.

    ratio: float
        The ratio at which the stepsize converges (0.5 - 1.0), default = 0.7.

    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """

    assert method in ['sgd', 'newton', 'gd'], "Argument 'method' must be either " + \
        ", ".join(f"'{x}'" for x in method)

    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic linear
    for iteration in range(max_iters):

        gamma_ = gamma
        if method in ["sgd", "newton"]:
            # Calculate gamma (Robbins-Monroe condition)
            gamma_ = gamma / pow(iteration + 1, ratio)

        # get loss and update w.
        loss, w, gradient = gradient_descent_step(
            y, tx, w, gamma_, lambda_, method=method)

        # log info
        if iteration % 1000 == 0:
            print("Current iteration={i}, loss={l}".format(
                i=iteration, l=loss))
            print("||d|| = {d}".format(d=np.linalg.norm(gradient)))
        # converge criterion
        losses.append(loss)
        # if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
        #    break

    # visualization
    print("loss={l}".format(l=compute_loss(y, tx, w)))

    return w, losses[-1]


def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                        max_iters: int, gamma: float, method: str = 'sgd', ratio: float = 0.7) -> Tuple[np.ndarray, float]:
    """
    Does the logistic linear.

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
        Gradient descent stepsize. Only used for gd.

    method: str
        The method for the optimization solution. Should be either SGD, Newton or GD.

    ratio: float
        The ratio at which the stepsize converges (0.5 - 1.0), default = 0.7.

    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """

    assert method in ['sgd', 'newton', 'gd'], "Argument 'method' must be either " + \
        ", ".join(f"'{x}'" for x in method)

    return reg_logistic_regression(y, tx, initial_w, 0, max_iters, gamma, method=method, ratio=ratio)


def gradient_descent_step(y: np.ndarray, tx: np.ndarray, w: np.ndarray, gamma: np.ndarray,
                          lambda_: float = 0, method='sgd') -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Does one step of gradient descent.

    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.

    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.

    w: ndarray
        Array containing the linear parameters to test.

    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.

    gamma: float
        The stepsize.

    method: str
        The method for the optimization solution. Should be either SGD, Newton or GD.
    Returns
    -------
    w: np.ndarray
        The linear parameters.

    loss: float
        The loss given w as parameters.
    """

    assert method in ['sgd', 'newton', 'gd'], "Argument 'method' must be either " + \
        ", ".join(f"'{x}'" for x in method)

    # Get loss, gradient, hessian
    loss = compute_loss(y, tx, w, lambda_=lambda_)

    rand_idx = randrange(y.shape[0])
    rand_tx = tx[rand_idx, :].reshape((-1, 1)).T
    rand_y = y[rand_idx]

    # Get the gradient of all the points (GD), or in a stochastic way (SGD, Newton)
    # We use this approach for Newton method to save memory.
    gradient = (compute_gradient(y, tx, w, lambda_=lambda_) if method == 'gd'
                else compute_gradient(rand_y, rand_tx, w, lambda_=lambda_))
    # Default case: w = w - gamma * g where g is the gradient
    g = gradient

    # In case the method is newton we need to compute the hessian
    if method == 'newton':
        hessian = compute_hessian(rand_y, rand_tx, w, lambda_=lambda_)
        g = (np.linalg.inv(hessian)).dot(gradient)

    # Update w
    w = w - gamma * g

    return loss, w, gradient
