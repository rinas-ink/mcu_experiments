from angles_dataset_generator import get_array_of_control_vars, get_array_of_figures
import numpy as np
from mcu import prepare_data, solve_semidefinite_programming, \
    get_eigen_decomposition, regress, plot_two_embeddings_3d, reduce_dimension


def maximum_covariance_unfolding_regression(control_vars, response_matrix):
    standardized_x, centered_y, edges, y_means, y_scaler, x_means, x_stds = prepare_data(control_vars, response_matrix)

    q = solve_semidefinite_programming(standardized_x, centered_y, edges)
    u, s = get_eigen_decomposition(q)
    m_ = 3
    y_ = reduce_dimension(u, s, m_)
    b = regress(y_, standardized_x)

    return standardized_x, centered_y, y_, b


def main(noise=True):
    control_vars = get_array_of_control_vars(noise=noise)
    response_matrix = get_array_of_figures(control_vars, noise=noise)
    standardized_x, centered_y, ld_embedding, regression_matrix = maximum_covariance_unfolding_regression(control_vars,
                                                                                                          response_matrix)
    reconstructed_y = np.dot(standardized_x, regression_matrix)

    plot_two_embeddings_3d(ld_embedding, reconstructed_y)


if __name__ == '__main__':
    main(False)

