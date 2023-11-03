from data import generate_array_of_swiss_rolls, get_control_vars, get_p
import numpy as np


def standardize(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / stds


def center(data):
    means = np.mean(data, axis=0)
    return data - means


def scaling(data):
    pass


def regress(y_, x):
    pass


def get_eigen_decomposition(q):
    pass


def solve_semi_definite_programming():
    pass


def compute(y, s):  # FIXME: fix name of the function
    pass


def maximum_covariance_unfolding_regression():
    control_vars = get_control_vars(get_p())
    response_matrix = generate_array_of_swiss_rolls(control_vars)

    control_vars = standardize(control_vars)
    response_matrix = center(response_matrix)
    response_matrix = scaling(response_matrix)

    q = solve_semi_definite_programming(response_matrix)
    u, s = get_eigen_decomposition(q)
    y_ = compute(u, s)
    b = regress(y_, control_vars)


if __name__ == '__main__':
    maximum_covariance_unfolding_regression()
