import numpy as np
from typing import Tuple

from src.logistic.cost import compute_loss
from src.logistic.gradient import compute_gradient
from src.logistic.hessian import compute_hessian

def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float, initial_w: np.ndarray,
                            max_iters: int, gamma: float, method: str ='sgd') -> Tuple[np.ndarray, float]:
    """
    Does the regularized logistic regression.
    
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.
    
    initial_w: ndarray
        Array containing the regression parameters to start with.
    
    max_iters: int
        The maximum number of iterations to do.
    
    gamma: float
        Gradient descent stepsize
        
    method: str
        THe method for the optimization solution. Should be either SGD, Newton or GD.

    Returns
    -------
    w: np.ndarray
        The regression parameters.
    
    loss: float
        The loss given w as parameters.
    """
    
    assert method is in ['sgd', 'newton', 'gd'] "method should be either 'sgd', 'newton' or 'gd'"

    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = gradient_descent_step(y, tx, w, gamma, lambda_, method=method)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    # visualization
    print("loss={l}".format(l=compute_loss(y, tx, w)))

    return w, losses[-1]

def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
        max_iters: int, gamma: float, method: str ='sgd') -> Tuple[np.ndarray, float]:
    """
    Does the logistic regression.
    
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    initial_w: ndarray
        Array containing the regression parameters to start with.
    
    max_iters: int
        The maximum number of iterations to do.
    
    gamma: float
        Gradient descent stepsize

    Returns
    -------
    w: np.ndarray
        The regression parameters.
    
    loss: float
        The loss given w as parameters.
    """
    
    assert method is in ['sgd', 'newton', 'gd'] "method should be either 'sgd', 'newton' or 'gd'"

    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma, method=method)

def gradient_descent_step(y: np.ndarray, tx: np.ndarray, w: np.ndarray, gamma:np.ndarray,
        lambda_: float = 0, method='sgd') -> Tuple[float, np.ndarray]:
    """
    Does one step of gradient descent.
    
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    
    w: ndarray
        Array containing the regression parameters to test.
    
    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.

    Returns
    -------
    w: np.ndarray
        The regression parameters.
    
    loss: float
        The loss given w as parameters.
    """
    
    assert method is in ['sgd', 'newton', 'gd'] "method should be either 'sgd', 'newton' or 'gd'"
    
    # TODO: implement different methods
    
    # Get loss, gradient, hessian
    loss = compute_loss(y, tx, w, lambda_=lambda_)
    gradient = compute_gradient(y, tx, w, lambda_=lambda_)
    hessian = compute_hessian(y, tx, w, lambda_=lambda_)

    # Update w
    w = w - gamma * (np.linalg.inv(hessian)).dot(gradient)
    
    return loss, w

    