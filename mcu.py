import dataset_generator
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
    stds[stds == 0] = 1  # To avoid division by 0
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
    # prob.solve(solver=cvxpy.SCS, verbose=True, \
    #            max_iters=1000000, acceleration_lookback=0, \
    #            adaptive_scale=adaptive_scale_sdp, scale=scale_sdp)
    prob.solve()
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


def plot_embeddings_vs_parameters(params, embedding, param_names=None, edges=None):
    _, params_cnt = np.shape(params)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(params_cnt)]
    fig, axs = plt.subplots(params_cnt * (params_cnt - 1) // 2, 2, figsize=(14, params_cnt * (params_cnt - 1) * 7 / 2))
    if axs.ndim == 1:
        axs = np.array([axs])
    cnt = 0
    for i in range(1, params_cnt):
        for j in range(i):
            axs[cnt, 0].scatter(params[:, j], params[:, i], s=10, c=params[:, j], cmap=plt.cm.Spectral)
            axs[cnt, 0].set_title('Params')
            axs[cnt, 0].set_xlabel(param_names[j])
            axs[cnt, 0].set_ylabel(param_names[i])

            axs[cnt, 1].scatter(embedding[:, j], embedding[:, i], s=10, c=embedding[:, j], cmap=plt.cm.Spectral)
            axs[cnt, 1].set_title('Embedding')
            axs[cnt, 1].set_xlabel(param_names[j])
            axs[cnt, 1].set_ylabel(param_names[i])
            if edges is None:
                axs[cnt, 1].set_xlim(1.1 * np.percentile(embedding[:, j], 10), 1.1 * np.percentile(embedding[:, j], 90))
                axs[cnt, 1].set_ylim(1.1 * np.percentile(embedding[:, i], 10), 1.1 * np.percentile(embedding[:, i], 90))
            else:
                edge_colors = ['red', 'green', 'blue']
                edge_colors = [edge_colors[np.random.randint(0, 3)] for _ in range(len(edges))]
                params_seg = np.hstack((params[edges[:, 0]][:, [j, i]], params[edges[:, 1]][:, [j, i]]))
                params_seg = params_seg.reshape((-1, 2, 2))
                rec_edges = LineCollection(params_seg, colors=edge_colors, alpha=0.5)
                axs[cnt, 0].add_collection(rec_edges)

                ld_segments = np.hstack((embedding[edges[:, 0]][:, [j, i]], embedding[edges[:, 1]][:, [j, i]]))
                ld_segments = ld_segments.reshape((-1, 2, 2))
                ld_edges = LineCollection(ld_segments, colors=edge_colors, alpha=0.5)
                axs[cnt, 1].add_collection(ld_edges)
            cnt += 1
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


def test_predictive_optimization(lw, up, p, k, figure_generator, figure_point_cnt,
                                 centered_y, ld_embedding, regression_matrix, y_means, y_scaler,
                                 x_stds, x_means, noise_level=0, pieces_cnt=10, test_data_size=50,
                                 same_value=False):
    intervals = [np.linspace(lw[i], up[i], pieces_cnt + 1) for i in range(p)]
    interval_runs_shape = tuple([pieces_cnt] * p + [p + 1, 2])
    interval_runs = np.empty(shape=interval_runs_shape)
    pieces_intervals = []

    for indices in np.ndindex(*[pieces_cnt] * p):
        interval_lw = np.array([intervals[dim][index] for dim, index in enumerate(indices)])
        interval_up = np.array([intervals[dim][index + 1] for dim, index in enumerate(indices)])

        if same_value:
            interval_lw = interval_up

        test_control_vars = dataset_generator.get_control_vars(deterministic=False,
                                                               dimensionality=p,
                                                               size=test_data_size,
                                                               lw=interval_lw, up=interval_up)
        test_rolls = dataset_generator.generate_array_of_figures(test_control_vars, figure_generator,
                                                                 noise_level=noise_level,
                                                                 min_num_points=figure_point_cnt)
        x_opts = []
        for (roll, control_var) in zip(test_rolls, test_control_vars):
            x_opt, x_err = predictive_optimization(roll, centered_y, ld_embedding, regression_matrix, y_means,
                                                   y_scaler, k)
            x_opt = x_opt * x_stds + x_means
            x_opts.append(x_opt)
            print("-----------")
            print(f"x_opt  = {x_opt}, x_err = {x_err}")
            print(f"x_real = {control_var}")

        x_opts = np.array(x_opts)
        test_control_vars = np.array(test_control_vars)
        errors = x_opts - test_control_vars
        errors_common = np.linalg.norm(errors, axis=1)

        interval_runs[indices] = [
                                     [np.median(abs(errors[:, dim])),
                                      np.percentile(abs(errors[:, dim]), 75) - np.percentile(abs(errors[:, dim]), 25)]
                                     for dim in range(p)
                                 ] + [[np.median(errors_common),
                                       np.percentile(errors_common, 75) - np.percentile(errors_common, 25)]]

        for dim in range(p):
            print(f'errors{dim} = {abs(errors[:, dim])}')

        pieces_intervals.append(np.column_stack((interval_lw, interval_up)).T)

    return interval_runs, np.array(pieces_intervals)


def plot_2d_predictive_optimization_heatmaps(lw, up, pieces_cnt, interval_runs, p, all_param_names=None,
                                             fixed_params_map=None, intervals = None):
    """

    :param lw: np.array of lower values for each parameter
    :param up: np.array of upper values for each parameter
    :param pieces_cnt: amount of pieces on which we cut each value range of parameters
    :param interval_runs: interval runs from `test_predictive_optimization`
    :param p: amount of parameters
    :param fixed_params_map: dictionary, that represents which parameter at which piece we fix.
    For example {0:1}, when p=3, means that we make this slice of `interval_runs`: `interval_runs[1, :, :]`.
    :param all_param_names: np.array of parameters names
    """
    if fixed_params_map is None:
        fixed_params_map = dict()
    if intervals is None:
        intervals = [np.linspace(lw[i], up[i], pieces_cnt + 1) for i in range(p)]
    if all_param_names is None:
        all_param_names = [f'param_{i}' for i in range(p)]
    fixed_indices = [slice(None)] * interval_runs.ndim
    for dim_index, value in fixed_params_map.items():
        fixed_indices[dim_index] = value
    interval_runs = interval_runs[tuple(fixed_indices)]

    remaining_params_idx = np.setdiff1d(np.arange(p), np.array(list(fixed_params_map.keys()), dtype=int))
    if len(remaining_params_idx) != 2:
        raise "There should be exactly 2 non-fixed parameters left"
    lw = lw[remaining_params_idx]
    up = up[remaining_params_idx]
    axes_param_names = all_param_names[remaining_params_idx]

    x_values = np.linspace(lw[0], up[0], pieces_cnt + 1)
    y_values = np.linspace(lw[1], up[1], pieces_cnt + 1)[::-1]

    fig, axs = plt.subplots(2, p + 1, figsize=(8 * (p + 1), 16))
    plot_title = "Fixed parameters: "
    for idx, piece in fixed_params_map.items():
        plot_title = plot_title + all_param_names[idx] + f' in [{intervals[piece][0][idx], intervals[piece][1][idx]}]; '

    fig.suptitle(plot_title, fontsize=16)

    imgs = []

    for l in range(2):
        for k in range(p + 1):
            imgs.append(axs[l, k].imshow(interval_runs[:, :, k, l], cmap='YlGnBu', interpolation='nearest'))
            axs[l, k].set_xlabel(axes_param_names[0])
            axs[l, k].set_ylabel(axes_param_names[1])
            axs[l, k].set_xticks(np.arange(pieces_cnt + 1) - 0.5, [f'{x:.1f}' for x in x_values])
            axs[l, k].set_yticks(np.arange(pieces_cnt + 1) - 0.5, [f'{y:.1f}' for y in y_values])
            fig.colorbar(imgs[-1], ax=axs[l, k])

            for i in range(pieces_cnt):
                for j in range(pieces_cnt):
                    axs[l, k].text(j, i, f'{interval_runs[i, j, k, l]:.1f}', ha='center', va='center', color='black')

    for k in range(p):
        axs[0, k].set_title(f'{all_param_names[k]} Error')
        axs[1, k].set_title(f'{all_param_names[k]} IQR')
    axs[0, p].set_title('Norm of common error')
    axs[1, p].set_title('Norm of common IQR')

    plt.tight_layout()
    plt.show()
