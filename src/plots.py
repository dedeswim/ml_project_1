# -*- coding: utf-8 -*-
"""Functions to plot figures."""
import numpy as np
import matplotlib.pyplot as plt


def plot_lambda_accuracy(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    plt.figure()
    plt.semilogx(lambdas, train_errors, color='b',
                 marker='*', label="Train accuracy")
    plt.semilogx(lambdas, test_errors, color='r',
                 marker='*', label="Test accuracy")
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("Ridge regression for quadratic expansion")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("lambda_search_accuracy")


def plot_lambda_error(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    plt.figure()
    plt.semilogx(lambdas, train_errors, color='b',
                 marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r',
                 marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for quadratic expansion")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("lambda_search_error")


def plot_poly_degree_error(train_errors, test_errors, degrees, lambda_):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    plt.figure()
    plt.plot(degrees, train_errors, color='b', marker='*', label="Train error")
    plt.plot(degrees, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("degree")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for lambda " + str(lambda_))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("poly_search_error")


def plot_poly_degree_accuracy(train_errors, test_errors, degrees, lambda_):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    plt.figure()
    plt.plot(degrees, train_errors, color='b',
             marker='*', label="Train accuracy")
    plt.plot(degrees, test_errors, color='r',
             marker='*', label="Test accuracy")
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("Ridge regression for lambda " + str(lambda_))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("poly_search_accuracy")
