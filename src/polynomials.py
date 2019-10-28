# -*- coding: utf-8 -*-
"""Contains functions to create polynomial expansions of the given x matrix"""

import numpy as np


def build_poly_matrix_vandermonde(x: np.ndarray, degree: int):
    """
    Homogeneous polynomial basis for input data x, for j=0 up to j=degree.
    """

    assert degree >= 2, "Argument 'degree' must be an integer >= 2"

    mat_van = np.polynomial.polynomial.polyvander(x.T, degree)
    return np.unique(np.hstack(list(mat_van[x] for x in range(mat_van.shape[0]))), axis=1)


def build_poly_matrix_quadratic(x: np.ndarray) -> np.ndarray:
    """
    Complete quadratic polynomial basis
    """

    cols = [c for c in x.T]
    cols.extend([c_1 * c_2 for i_1, c_1 in enumerate(cols)
                 for c_2 in cols[i_1:]])
    return np.column_stack(cols)
