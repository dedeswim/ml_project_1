import numpy as np
from costs import compute_loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = compute_loss(y, tx, w)
    
    return mse, w