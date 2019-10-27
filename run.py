import numpy as np
import time

from src.helpers import load_csv_data, standardize, predict_labels, get_jet_indexes, jet_indexes, compute_accuracy, log_indexes, create_csv_submission
from implementations import ridge_regression
from src.polynomials import build_poly_matrix_quadratic
from src.plots import plot_lambda_accuracy, plot_lambda_error, plot_poly_degree_accuracy, plot_poly_degree_error
from src.linear.loss import compute_loss
from src.k_fold import cross_validation, build_k_indices

def ridge_regression_sets(x, y, lambda_, k):
    
    # Create lists to save the different ws, accuracies and losses for the
    # different subsets we run the training on
    ws = []
    te_accs = []
    tr_accs = []
    te_losses = []
    tr_losses = []

    # Iterate over the different subsets
    for i in x_jet_indexes:
        
        # Get the rows relative to the i-th subset taken in consideration
        tx_i = x[x_jet_indexes[i]]
        y_i = y[x_jet_indexes[i]]
        
        # Delete the columns that are -999 for the given subset
        tx_del = np.delete(tx_i, jet_indexes[i], axis=1)
        
        # Take the logarithm of each column
        for li in range(tx_del.shape[1]):
            tx_del[:,li] = np.apply_along_axis(lambda n: np.log(1 + abs(tx_del[:,li].min()) + n), 0, tx_del[:,li])
        
        # Standardize the data
        tx_std = standardize(tx_del)[0]
        
        # Build the polynomial expansion of degree 2 and add the 1s column
        tx = build_poly_matrix_quadratic(tx_std)
        tx = np.c_[np.ones((y_i.shape[0], 1)), tx]
        
        # Get indices for cross-validation
        k_indices = build_k_indices(y_i, k, 1)
        
        # Perform the training on the given-subset using cross validation and Ridge Regression
        w, tr_loss, te_loss, tr_acc, te_acc = \
            cross_validation(y_i, tx, k_indices, k, lambda_, ridge_regression)

        # Add the results of the training of the i-th subset
        ws.append(w)
        te_accs.append(te_acc * tx_i.shape[0])
        tr_accs.append(tr_acc * tx_i.shape[0])
        te_losses.append(te_loss * tx_i.shape[0])
        tr_losses.append(tr_loss * tx_i.shape[0])
    
    # Compute the mean results from all the subsets
    mean_tr_loss = sum(tr_losses) / x.shape[0]
    mean_te_loss = sum(te_losses) / x.shape[0]
    mean_tr_acc = sum(tr_accs) / x.shape[0]
    mean_te_acc = sum(te_accs) / x.shape[0]
    
    return ws, mean_tr_loss, mean_te_loss, mean_tr_acc, mean_te_acc 

# Import data
y, x_raw, ids = load_csv_data('data/train.csv')

# get the i-th subset
x_jet_indexes = get_jet_indexes(x_raw)
x = x_raw
x_mass = np.zeros(x.shape[0])
x_mass[x[:, 0] == -999] = 1
x[:, 0][x[:, 0] == -999] = np.median(x[:, 0][x[:, 0] != -999])
x = np.column_stack((x, x_mass))

lambda_ = 1e-5
k = 5

print("Starting training with Ridge Regression...\n")
ws, tr_loss, te_loss, tr_acc, te_acc = ridge_regression_sets(x, y, lambda_, k)

print("Train accuracy={tr_acc:.3f}, test accuracy={te_acc:.3f}".format(tr_acc=tr_acc, te_acc=te_acc))
print("Train MSE={tr_loss:.3f}, test MSE={te_loss:.3f}".format(tr_loss=tr_loss, te_loss=te_loss))

print("\n\nGenerating .csv file...")

y_sub, x_sub_raw, ids_sub = load_csv_data('data/test.csv')
x_sub_jet_indexes = get_jet_indexes(x_sub_raw)

x_sub = x_sub_raw

x_sub_mass = np.zeros(x_sub.shape[0])
x_sub_mass[x_sub[:, 0] == -999] = 1
x_sub[:, 0][x_sub[:, 0] == -999] = np.median(x_sub[:, 0][x_sub[:, 0] != -999])
x_sub = np.column_stack((x_sub, x_sub_mass))

for i, w in enumerate(ws):
    
    tx_sub_i = x_sub[x_sub_jet_indexes[i]] 
    tx_sub_del = np.delete(tx_sub_i, jet_indexes[i], axis=1)
    
    for li in range(tx_sub_del.shape[1]):
        # print(tx_del[:,li].min())
        tx_sub_del[:,li] = np.apply_along_axis(lambda n: np.log(1 + abs(tx_sub_del[:,li].min()) + n), 0, tx_sub_del[:,li])
    
    tx_sub_std = standardize(tx_sub_del)[0]
    tx_sub_poly = build_poly_matrix_quadratic(tx_sub_std)
    
    tx_sub = np.c_[np.ones((y_sub[x_sub_jet_indexes[i]].shape[0], 1)), tx_sub_poly]
    
    y_sub[x_sub_jet_indexes[i]] = predict_labels(ws[i], tx_sub, mode='linear')
    
create_csv_submission(ids_sub, y_sub, 'final-test.csv')

print("\nfinal-test.csv file generated")