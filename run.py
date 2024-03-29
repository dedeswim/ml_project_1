import numpy as np
import time

from src.helpers import load_csv_data, predict_labels, get_jet_indexes, \
    create_csv_submission, clean_mass_feature, prepare_x
from implementations import ridge_regression
from src.k_fold import cross_validation, build_k_indices


def ridge_regression_sets(x, y, lambda_, k):
    # Create lists to save the different ws, accuracies and losses for the
    # different subsets we run the training on
    ws = []
    te_accs = []
    tr_accs = []
    te_losses = []
    tr_losses = []

    # Get the mask with the rows belonging to each subset
    x_jet_indexes = get_jet_indexes(x)

    # Iterate over the different subsets
    for i, indexes in enumerate(x_jet_indexes):
        # Get the rows relative to the i-th subset taken in consideration
        tx_i = prepare_x(x, x_jet_indexes, i)
        y_i = y[indexes]

        # Get indices for cross-validation
        k_indices = build_k_indices(y_i, k, 1)

        # Perform the training on the given-subset using cross validation and Ridge Regression
        w, tr_loss, te_loss, tr_acc, te_acc = \
            cross_validation(y_i, tx_i, k_indices, k,
                             lambda_, ridge_regression)

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


def main():
    # Import data
    y, x_raw, ids = load_csv_data('data/train.csv')

    # Deal with mass missing values
    x = clean_mass_feature(x_raw)

    # Set hyperparameters
    lambda_ = 1e-5
    k = 5

    print("Starting training with Ridge Regression...\n")

    # Run Ridge Regression
    ws, tr_loss, te_loss, tr_acc, te_acc = ridge_regression_sets(
        x, y, lambda_, k)

    print("Train accuracy={tr_acc:.3f}, test accuracy={te_acc:.3f}".format(
        tr_acc=tr_acc, te_acc=te_acc))
    print("Train MSE={tr_loss:.3f}, test MSE={te_loss:.3f}".format(
        tr_loss=tr_loss, te_loss=te_loss))

    print("\n\nGenerating .csv file...")

    # Open the test dataset
    y_sub, x_sub_raw, ids_sub = load_csv_data('data/test.csv')

    # Get the mask with the rows belonging to each subset
    x_sub_jet_indexes = get_jet_indexes(x_sub_raw)

    # Deal with mass missing values
    x_sub = clean_mass_feature(x_sub_raw)

    # Iterate over the parameters of the models trained on the different
    # subsets, and predict the labels of each subset
    for i, w in enumerate(ws):
        # Prepare the feature of the i-th subset
        tx_sub = prepare_x(x_sub, x_sub_jet_indexes, i)

        # Predict the labels
        y_sub[x_sub_jet_indexes[i]] = predict_labels(
            ws[i], tx_sub, mode='linear')

    # Create the submission file
    create_csv_submission(ids_sub, y_sub, 'final-test.csv')

    print("\nfinal-test.csv file generated")


if __name__ == "__main__":
    main()
