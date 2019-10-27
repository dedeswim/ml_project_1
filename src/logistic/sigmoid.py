import numpy as np
from typing import Union


def sigmoid(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Element-wise applies sigmoid function on t.

    Parameters
    ----------
    t: float or ndarray
        Float or array onto which the sigmoid needs to be applied.

    Returns
    -------
    s(t): float or ndarray
        Float or array to which the sigmoid function has been applied.
    """

    return 1 / (1 + np.exp(-t))
