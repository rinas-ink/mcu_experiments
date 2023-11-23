from angles_dataset_generator import create_figure, get_p, get_array_of_control_vars, get_array_of_figures
import matplotlib.pyplot as plt
import numpy as np
import cvxpy

N = 50


def standardize(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / stds


def center(data):
    means = np.mean(data, axis=0)
    return data - means


def scale(data):
    scaler = np.average(np.sqrt(np.var(data, axis=0)))
    return data / scaler


def construct_graph(ys, k):
    edges = np.empty((0, 2), dtype=int)
    for y in ys:
        distances = np.linalg.norm(ys - y, axis=1)
        neighbours = np.argsort(distances)[:k]
        all_pairs = np.stack(np.meshgrid(neighbours, neighbours), axis=1).reshape(-1, 2)
        edges = np.vstack((edges, all_pairs))
    return np.unique(edges, axis=0)


def solve_semidefinite_programming(xs, ys, edges, c):
    n = xs.shape[0]
    p = np.dot(ys, ys.T)
    q = cvxpy.Variable((n, n), symmetric=True)

    constraints = [q >> 0]
    constraints += [cvxpy.trace(np.ones((n, n)) @ q) == 0]
    constraints += [cvxpy.trace(q) <= (n - 1) * c]
    constraints += [
        q[i][i] + q[j][j] - 2 * q[i][j] == p[i][i] + p[j][j] - 2 * p[i][j] for i, j in edges
    ]

    prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.trace(xs @ xs.T @ q)), constraints)
    prob.solve()
    return q.value


def get_eigen_decomposition(q):
    eigenvalues, eigenvectors = np.linalg.eig(q)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvectors, np.diag(sorted_eigenvalues)


def reduce_dimension(eigenvectors, sigma):  # FIXME: add Otsu's threshold
    # m_ = 1
    # while np.sum(sigma[:, :m_]) < 0.9 * np.sum(sigma):
    #     m_ += 1
    m_ = 2
    return np.dot(eigenvectors[:, :m_], np.sqrt(sigma[:m_, :m_]))


def regress(y_, x):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y_)


def maximum_covariance_unfolding_regression(control_vars, response_matrix):
    control_vars = standardize(control_vars)
    response_matrix = center(response_matrix)
    response_matrix = scale(response_matrix)

    k = 5  # FIXME
    c = 1e5  # FIXME

    edges = construct_graph(response_matrix, k)
    q = solve_semidefinite_programming(control_vars, response_matrix, edges, c)
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
    control_vars = get_array_of_control_vars(get_p(), N)
    initial_point = (0, 0, 0)
    response_matrix = get_array_of_figures(control_vars, N)
    standardized_x, centered_y, ld_embedding, regression_matrix = maximum_covariance_unfolding_regression(control_vars,
                                                                                                          response_matrix)
    reconstructed_y = np.dot(standardized_x, regression_matrix)

    print(compute_rre(ld_embedding, reconstructed_y))
    plot_two_embeddings(ld_embedding, reconstructed_y)


if __name__ == '__main__':
    main()
