import matplotlib.pyplot as plt
import numpy as np
import cvxpy
from scipy.optimize import dual_annealing


def standardize(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / stds, means, stds


def center(data):
    means = np.mean(data, axis=0)
    return data - means, means


def scale(data):
    scaler = np.average(np.sqrt(np.var(data, axis=0)))
    return data / scaler, scaler


def construct_graph(ys, k):
    edges = np.empty((0, 2), dtype=int)
    for y in ys:
        distances = np.linalg.norm(ys - y, axis=1)
        neighbours = np.argsort(distances)[:k]
        all_pairs = np.stack(np.meshgrid(neighbours, neighbours), axis=1).reshape(-1, 2)
        edges = np.vstack((edges, all_pairs))
    return np.unique(edges, axis=0)


def solve_semidefinite_programming(xs, ys, edges):
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


def choose_principal_branches(eigenvectors, sigma, m_=2):  # FIXME: add Otsu's threshold
    # m_ = 1
    # while np.sum(sigma[:, :m_]) < 0.9 * np.sum(sigma):
    #     m_ += 1

    return np.dot(eigenvectors[:, :m_], np.sqrt(sigma[:m_, :m_]))


def regress(y_, x):
    B, _, _, _ = np.linalg.lstsq(x.T.dot(x), x.T.dot(y_), rcond=None)
    return B


#   return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y_)


k = 5  # FIXME
c = 1e5  # FIXME


def prepare_data(control_vars, response_matrix):
    control_vars, x_means, x_stds = standardize(control_vars)
    response_matrix, y_means = center(response_matrix)
    response_matrix, y_scaler = scale(response_matrix)

    edges = construct_graph(response_matrix, k)
    return control_vars, response_matrix, edges, y_means, y_scaler, x_means, x_stds


def reduce_dimensions(q):
    u, s = get_eigen_decomposition(q)
    y_ = choose_principal_branches(u, s)
    return y_


def compute_rre(ld_embedding, reconstructed_y):
    return np.linalg.norm(ld_embedding - reconstructed_y, axis=1) / np.linalg.norm(ld_embedding, axis=1)


def plot_rre_heatmap(rre, reconstructed_y):
    fig = plt.figure(figsize=(6, 6))
    scatter = plt.scatter(reconstructed_y[:, 0], reconstructed_y[:, 1], s=20, c=rre, cmap='viridis', edgecolors='w',
                          vmin=0, vmax=1)
    cbar = plt.colorbar(scatter)
    plt.show()


def plot_two_embeddings(ld_embedding, reconstructed_y):
    fig = plt.figure(figsize=(14, 7))
    rec_plot = fig.add_subplot(1, 2, 2)
    rec_plot.scatter(reconstructed_y[:, 0], reconstructed_y[:, 1], s=10, c=reconstructed_y[:, 0], cmap=plt.cm.Spectral)

    ld_plot = fig.add_subplot(1, 2, 1)
    ld_plot.set_xlim(rec_plot.get_xlim())
    ld_plot.set_ylim(rec_plot.get_ylim())
    ld_plot.scatter(ld_embedding[:, 0], ld_embedding[:, 1], s=10, c=ld_embedding[:, 0], cmap=plt.cm.Spectral)

    plt.show()


def predictive_optimization(y_nom, centered_y, ld_embedding, regression_matrix, y_means, y_scaler):
    y_nom = (y_nom - y_means) / y_scaler
    distances = np.linalg.norm(centered_y - y_nom, axis=1)
    neighbours = np.argsort(distances)[:k]

    def y_error(v):
        err_diff = (np.linalg.norm(v - ld_embedding[neighbours]) -
                    np.linalg.norm(y_nom - centered_y[neighbours]))
        sum_err = np.sum(err_diff ** 2)
        return sum_err

    def x_error(x):
        return y_error(np.dot(x, regression_matrix))

    lw = [-1.3] * np.shape(regression_matrix)[1]
    up = [1.3] * np.shape(regression_matrix)[1]

    x_opt = dual_annealing(x_error, bounds=list(zip(lw, up)))
    return x_opt.x, x_error(x_opt.x)
