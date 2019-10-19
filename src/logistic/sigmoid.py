import numpy as np
from src.logistic.sigmoid import sigmoid

def sigmoid(t: [float | np.ndarray]) -> [float | np.ndarray]:
    """
    Element-wise applies sigmoid function on t.
    
    Parameters
    ----------
    y: float | ndarray
        Float or array onto which the sigmoid needs to be applied.

    Returns
    -------
    s(t): float | ndarray
        Float or array to which the sigmoid function has been applied.
    """

    return 1 / (1 + np.exp(-t))