def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    lambda_p = lambda_ * 2 * tx.shape[0]
    
    w_ridge = np.linalg.solve(tx.T.dot(tx) + lambda_p * np.eye(tx.shape[1]), tx.T.dot(y))
    mse = compute_loss(y, tx, w_ridge)
    
    return mse, w_ridge