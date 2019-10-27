import numpy as np
from src.polynomials import build_poly_matrix_vandermonde
from src.linear.loss import compute_loss
from src.helpers import compute_accuracy


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, model, mean=True):
    """return the loss of ridge linear."""
    # Get k'th subgroup in test, others in train

    losses_tr, losses_te, accs_tr, accs_te, ws = [], [], [], [], []

    for k_ in range(k):
        test_indices = k_indices[k_]
        train_indices = np.setdiff1d(k_indices.flatten(), test_indices)

        y_train = y[train_indices]
        x_train = x[train_indices]
        y_test = y[test_indices]
        x_test = x[test_indices]

        # Ridge linear
        loss_tr, w = model(y_train, x_train, lambda_)

        # Calculate the loss for test data
        loss_te = compute_loss(y_test, x_test, w)
        
        acc_tr = compute_accuracy(x_train, w, y_train, mode='linear')
        acc_te = compute_accuracy(x_test, w, y_test, mode='linear')
        
        accs_te.append(acc_te)
        accs_tr.append(acc_tr)

        losses_tr.append(np.math.sqrt(2 * loss_tr))
        losses_te.append(np.math.sqrt(2 * loss_te))
        ws.append(w)

    return np.mean(ws, axis=0), np.mean(losses_tr), np.mean(losses_te), np.mean(accs_tr), np.mean(accs_te)
