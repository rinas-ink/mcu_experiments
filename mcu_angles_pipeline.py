from angles_dataset_generator import get_set_of_control_vars, get_array_of_figures
import matplotlib.pyplot as plt
import numpy as np
from mcu import prepare_data, construct_graph, get_k, solve_semidefinite_programming, \
    get_eigen_decomposition, get_optimal_dimensionality


def regress(y_, x):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y_)


def reduce_dimension(eigenvectors, sigma):  # FIXME: add Otsu's threshold
    # m_ = 1
    # while np.sum(sigma[:, :m_]) < 0.9 * np.sum(sigma):
    #     m_ += 1
    m_ = 2
    return np.dot(eigenvectors[:, :m_], np.sqrt(sigma[:m_, :m_]))


def maximum_covariance_unfolding_regression(control_vars, response_matrix):
    standardized_x, centered_y, edges, y_means, y_scaler, x_means, x_stds = prepare_data(control_vars, response_matrix)
    k = get_k()
    edges = construct_graph(response_matrix, k)
    q = solve_semidefinite_programming(standardized_x, centered_y, edges)
    u, s = get_eigen_decomposition(q)

    y_ = reduce_dimension(u, s)
    b = regress(y_, control_vars)

    return control_vars, response_matrix, y_, b


def compute_rre(ld_embedding, reconstructed_y):
    return np.median(np.linalg.norm(ld_embedding - reconstructed_y, axis=1) / np.linalg.norm(ld_embedding, axis=1))


def plot_two_embeddings(ld_embedding, reconstructed_y):
    n = ld_embedding.shape[0]
    fig = plt.figure(figsize=(14, 14))
    ld_plot = fig.add_subplot(1, 2, 1)
    ld_plot.scatter(ld_embedding[:, 0], ld_embedding[:, 1], s=10, c=ld_embedding[:, 0], cmap=plt.cm.Spectral)
    rec_plot = fig.add_subplot(1, 2, 2)
    rec_plot.scatter(reconstructed_y[:, 0], reconstructed_y[:, 1], s=10, c=reconstructed_y[:, 0], cmap=plt.cm.Spectral)
    plt.show()


def main():
    control_vars = get_set_of_control_vars()
    response_matrix = get_array_of_figures(control_vars)
    standardized_x, centered_y, ld_embedding, regression_matrix = maximum_covariance_unfolding_regression(control_vars,
                                                                                                          response_matrix)
    reconstructed_y = np.dot(standardized_x, regression_matrix)

    print(compute_rre(ld_embedding, reconstructed_y))
    plot_two_embeddings(ld_embedding, reconstructed_y)


if __name__ == '__main__':
    main()
