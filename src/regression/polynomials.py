import numpy as np

def build_poly_matrix(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    # TOFIX

    repetitions = np.ones(len(x.shape) + 1, dtype=int)
    repetitions[0] = degree + 1

    repeated_x = np.tile(x, repetitions)
    powers = np.mgrid[0:repeated_x.shape[0], 0:repeated_x.shape[1], 0:repeated_x.shape[2]][0]
    
    return np.power(repeated_x, powers).flatten().reshape(-1 , x.shape[1] * (degree + 1))

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    return np.power(x.repeat(degree + 1).reshape(x.shape[0], -1), np.arange(degree + 1))


def build_poly_matrix_vandermonde(x, degree):
    """
    TOWRITE
    """

    mat_van = np.polynomial.polynomial.polyvander(x.T, degree)
    return np.hstack(list(mat_van[x] for x in range(mat_van.shape[0])))