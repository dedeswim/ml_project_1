import numpy as np

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree, model, mean=True):
    """return the loss of ridge regression."""
    # Get k'th subgroup in test, others in train
    
    losses_tr, losses_te, ws = [], [], []
    
    for k_ in range(k):
        
        test_indices = k_indices[k_]
        train_indices = np.setdiff1d(k_indices.flatten(), test_indices)

        y_train = y[train_indices]
        x_train = x[train_indices]
        y_test = y[test_indices]
        x_test = x[test_indices]

        # Form data with polynomial degree
        x_train_poly = build_poly(x_train, degree)
        x_test_poly = build_poly(x_test, degree)

        # Ridge regression
        loss_tr, w = model(y_train, x_train_poly, lambda_)

        # Calculate the loss for test data
        loss_te = compute_loss(y_test, x_test_poly, w)
        
        losses_tr.append(np.math.sqrt(2 * loss_tr))
        losses_te.append(np.math.sqrt(2 * loss_te))
        ws.append(w_ridge)
    
        
        
    return np.mean(losses_tr), np.mean(losses_te)