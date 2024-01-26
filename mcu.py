import matplotlib.pyplot as plt
import numpy as np
import cvxpy
from matplotlib.collections import LineCollection
from itertools import combinations
from scipy.optimize import dual_annealing
from skimage.filters import threshold_otsu


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


def get_k():
    return 5


def get_c():
    return 1e5


# Returns indices of k nearest neighbours of y in ys and y itself
def k_nearest_neighbours(ys, y, k):
    distances = np.round(np.linalg.norm(ys - y, axis=1), 12)
    neighbours = np.argsort(distances, kind='stable')[:k + 1]
    return neighbours


def construct_graph(ys, k):
    assert (0 < k < len(ys))
    edges = np.empty((0, 2), dtype=int)
    for y in ys:
        neighbours = k_nearest_neighbours(ys, y, k)
        all_pairs = np.array(list(combinations(sorted(neighbours), 2)))
        edges = np.vstack((edges, all_pairs))
    return np.unique(edges, axis=0)

def solve_semidefinite_programming(xs, ys, edges, c=get_c(), adaptive_scale_sdp=True, scale_sdp=0.1):
    n = xs.shape[0]
    p = np.dot(ys, ys.T)
    q = cvxpy.Variable((n, n), PSD=True)
    constraints = [cvxpy.trace(np.ones((n, n)) @ q) == 0]
    constraints += [cvxpy.trace(q) <= (n - 1) * c]
    constraints += [
        q[i][i] + q[j][j] - 2 * q[i][j] == p[i][i] + p[j][j] - 2 * p[i][j] for i, j in edges
    ]

    prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.trace(xs @ xs.T @ q)), constraints)
    prob.solve(solver=cvxpy.SCS, verbose=True, \
                max_iters=1000000, acceleration_lookback=0, \
                adaptive_scale=adaptive_scale_sdp, scale=scale_sdp)
    return q.value


def get_eigen_decomposition(q):
    eigenvalues, eigenvectors = np.linalg.eig(q)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvectors, np.diag(sorted_eigenvalues)


def get_optimal_dimensionality(sigma):
    eigenvalues = np.diagonal(sigma)
    eigenvalues = eigenvalues[eigenvalues > 0]
    log_of_eigenvalues = np.log(eigenvalues)
    threshold = threshold_otsu(log_of_eigenvalues)
    m_ = 0
    while m_ < len(log_of_eigenvalues) and log_of_eigenvalues[m_] > threshold:
        m_ += 1
    return m_


def reduce_dimension(eigenvectors, sigma, m_):
    return np.dot(eigenvectors[:, :m_], np.sqrt(sigma[:m_, :m_]))


def regress(y_, x):
    B, _, _, _ = np.linalg.lstsq(x.T.dot(x), x.T.dot(y_), rcond=None)
    return B


def prepare_data(control_vars, figures, k=get_k()):
    control_vars, x_means, x_stds = standardize(control_vars)
    figures, y_means = center(figures)
    figures, y_scaler = scale(figures)

    edges = construct_graph(figures, k)
    return control_vars, figures, edges, y_means, y_scaler, x_means, x_stds


def reduce_dimensions(q, m_):
    u, s = get_eigen_decomposition(q)
    y_ = reduce_dimension(u, s, m_)
    return y_


def compute_rre_median(ld_embedding, reconstructed_y):
    return np.median(np.linalg.norm(ld_embedding - reconstructed_y, axis=1) / np.linalg.norm(ld_embedding, axis=1))


def compute_rre(ld_embedding, reconstructed_y):
    return np.linalg.norm(ld_embedding - reconstructed_y, axis=1) / np.linalg.norm(ld_embedding, axis=1)


def diff_of_edges_lengths(ld_embedding, reconstructed_y, edges):
    edge_lengths_ld = np.linalg.norm(ld_embedding[edges[:, 0]] - ld_embedding[edges[:, 1]], axis=1)
    edge_lengths_rec = np.linalg.norm(reconstructed_y[edges[:, 0]] - reconstructed_y[edges[:, 1]], axis=1)
    return edge_lengths_ld - edge_lengths_rec


def plot_rre_heatmap(rre, reconstructed_y):
    fig = plt.figure(figsize=(6, 6))
    scatter = plt.scatter(reconstructed_y[:, 0], reconstructed_y[:, 1], s=20, c=rre, cmap='viridis', edgecolors='w',
                          vmin=0)
    cbar = plt.colorbar(scatter)
    plt.show()


def plot_two_embeddings_3d(ld_embedding, reconstructed_y):
    slices = [slice(None, 2), slice(1, None), [0, -1]]
    n_slices = len(slices)

    fig, axs = plt.subplots(n_slices, 2, figsize=(15, 5 * n_slices))

    for i, sl in enumerate(slices):
        ld_emb_slice = ld_embedding[:, sl]
        rec_y_slice = reconstructed_y[:, sl]
        axs[i, 0].scatter(ld_emb_slice[:, 0], ld_emb_slice[:, 1], s=10, c=ld_emb_slice[:, 0], cmap=plt.cm.Spectral)
        axs[i, 1].scatter(rec_y_slice[:, 0], rec_y_slice[:, 1], s=10, c=rec_y_slice[:, 0], cmap=plt.cm.Spectral)

        rre = compute_rre_median(ld_emb_slice, rec_y_slice)

        axs[i, 0].set_title(f'ld_embedding (slice {i + 1}), RRE: {rre:.6f}')
        axs[i, 1].set_title(f'reconstructed_y (slice {i + 1}), RRE: {rre:.6f}')

    plt.show()


def plot_embeddings_vs_parameters(ld_embedding, reconstructed_y):
    fig = plt.figure(figsize=(14, 7))

    rec_plot = fig.add_subplot(1, 2, 2)
    rec_plot.scatter(reconstructed_y[:, 0], reconstructed_y[:, 1], s=10, c=reconstructed_y[:, 0], cmap=plt.cm.Spectral)
    rec_plot.set_title('Reconstructed Embedding')

    ld_plot = fig.add_subplot(1, 2, 1)
    ld_plot.scatter(ld_embedding[:, 0], ld_embedding[:, 1], s=10, c=ld_embedding[:, 0], cmap=plt.cm.Spectral)
    ld_plot.set_title('Params')
    ld_plot.set_xlim(rec_plot.get_xlim())
    ld_plot.set_ylim(rec_plot.get_ylim())

    plt.show()


def compute_3d_rre_median(ld_embedding, reconstructed_y):
    slices = [slice(None, 2), slice(1, None), [0, -1]]
    rre_arr = []

    for i, sl in enumerate(slices):
        ld_emb_slice = ld_embedding[:, sl]
        rec_y_slice = reconstructed_y[:, sl]
        rre_arr.append(compute_rre_median(ld_emb_slice, rec_y_slice))

    return np.array(rre_arr)


def plot_two_embeddings_with_edges(ld_embedding, reconstructed_y, edges):
    fig = plt.figure(figsize=(14, 7))
    rec_plot = fig.add_subplot(1, 2, 2)
    rec_plot.scatter(reconstructed_y[:, 0], reconstructed_y[:, 1], s=25, c=reconstructed_y[:, 0], cmap=plt.cm.Spectral)

    ld_plot = fig.add_subplot(1, 2, 1)
    ld_plot.set_xlim(rec_plot.get_xlim())
    ld_plot.set_ylim(rec_plot.get_ylim())
    ld_plot.scatter(ld_embedding[:, 0], ld_embedding[:, 1], s=25, c=ld_embedding[:, 0], cmap=plt.cm.Spectral)
    for i, j in edges:
        rec_plot.plot([reconstructed_y[i, 0], reconstructed_y[j, 0]], [reconstructed_y[i, 1], reconstructed_y[j, 1]],
                      color='black', linestyle='-', linewidth=1)
        ld_plot.plot([ld_embedding[i, 0], ld_embedding[j, 0]], [ld_embedding[i, 1], ld_embedding[j, 1]], color='black',
                     linestyle='-', linewidth=1)
    plt.show()


def plot_graph(edges, ld_embedding, reconstructed_y):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    edge_colors = ['red', 'green', 'blue']
    edge_colors = [edge_colors[np.random.randint(0, 2)] for _ in range(len(edges))]

    rec_plot_graph = axes[1]
    rec_plot_graph.scatter(reconstructed_y[:, 0], reconstructed_y[:, 1], s=20, c=reconstructed_y[:, 0],
                           cmap=plt.cm.Spectral)
    rec_segments = np.hstack((reconstructed_y[edges[:, 0]], reconstructed_y[edges[:, 1]]))
    rec_segments = rec_segments.reshape((-1, 2, 2))
    rec_edges = LineCollection(rec_segments, colors=edge_colors, alpha=0.5)
    rec_plot_graph.add_collection(rec_edges)

    ld_plot_graph = axes[0]
    ld_plot_graph.scatter(ld_embedding[:, 0], ld_embedding[:, 1], s=20, c=ld_embedding[:, 0], cmap=plt.cm.Spectral)

    ld_segments = np.hstack((ld_embedding[edges[:, 0]], ld_embedding[edges[:, 1]]))
    ld_segments = ld_segments.reshape((-1, 2, 2))
    ld_edges = LineCollection(ld_segments, colors=edge_colors, alpha=0.5)
    ld_plot_graph.add_collection(ld_edges)

    plt.show()


def predictive_optimization(y_nom, centered_y, ld_embedding, regression_matrix, y_means, y_scaler, k, seed=-1):

    y_nom = (y_nom - y_means) / y_scaler
    distances = np.linalg.norm(centered_y - y_nom, axis=1)
    neighbours = np.argsort(distances)[:k]

    def y_error(v):
        err_diff = (np.linalg.norm(v - ld_embedding[neighbours], axis=1) -
                    np.linalg.norm(y_nom - centered_y[neighbours], axis=1))
        sum_err = np.sum(err_diff ** 2)
        return sum_err

    def x_error(x):
        return y_error(np.dot(x, regression_matrix))

    lw = [-1.3] * np.shape(regression_matrix)[1]
    up = [1.3] * np.shape(regression_matrix)[1]

    if seed == -1:
        x_opt = dual_annealing(x_error, bounds=list(zip(lw, up)))
    else:
        x_opt = dual_annealing(x_error, bounds=list(zip(lw, up)), seed=seed)
    return x_opt.x, x_error(x_opt.x)

# def plot_predictive_optimization_error(x_opt, x_real):
