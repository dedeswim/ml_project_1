import numpy as np


def build_poly_matrix(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    repetitions = np.ones(len(x.shape) + 1, dtype=int)
    repetitions[0] = degree + 1

    repeated_x = np.tile(x, repetitions)
    powers = np.mgrid[0:repeated_x.shape[0], 0:repeated_x.shape[1], 0:repeated_x.shape[2]][0]

    return np.power(repeated_x, powers).flatten().reshape(-1, x.shape[1] * (degree + 1))


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    return np.power(x.repeat(degree + 1).reshape(x.shape[0], -1), np.arange(degree + 1))


def build_poly_matrix_vandermonde(x, degree):
    """
    TOWRITE
    """

    mat_van = np.polynomial.polynomial.polyvander(x.T, degree)
    return np.unique(np.hstack(list(mat_van[x] for x in range(mat_van.shape[0]))), axis=1)

def build_poly_matrix_quadratic(x):
    """
    TOWRITE
    """

    cols = [c for c in x.T]
    cols.extend([c_1 * c_2 for i_1,c_1 in enumerate(cols) for c_2 in cols[i_1:]])
    return np.column_stack(cols)