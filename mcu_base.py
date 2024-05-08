import time
from abc import abstractmethod

from matplotlib import pyplot as plt
from scipy.optimize import dual_annealing, minimize

import auxiliary
import numpy as np
import cvxpy
from itertools import combinations


class MCUbase:
    def __init__(self, k, c, figures, params, params_names=None):
        # TODO remove params dimensions that are almost constant
        assert (0 < k < len(figures))
        assert (params.shape[0] == len(figures))
        self.k = k
        self.c = c
        self.original_params = params
        self.params_dim = params.shape[1]
        self.params_names = params_names
        self.params = None
        self.params_means = None
        self.params_stds = None
        self.params_lw = None
        self.params_up = None
        self.original_figures = figures
        self.figures = None
        self.figures_means = None
        self.figures_scaler = None
        self.graph_edges = None
        self.q = None
        self.embedded_y = None
        self.B = None
        self.figures_count = params.shape[0]
        self.dists = None
        self.k_neighbors = None
        self.avg_k = 0
        self.prob = None
        self.dists_scaler = None

    @abstractmethod
    def k_nearest_neighbours(self, y, k=None, symmetric=False):
        pass

    def train(self, max_iters=100, do_cliques=False, keep_mutual_only=True):
        self.prepare_data()
        self.construct_graph(do_cliques, keep_mutual_only)
        self.solve_semidefinite_programming(max_iters)
        self.reduce_dimensions()
        self.embedded_y_to_param_regression()

    def prepare_data(self):
        if self.params is not None:
            return
        self.params, self.params_means, self.params_stds = auxiliary.standardize(self.original_params)
        self.center_and_scale_figures()

    @abstractmethod
    def center_and_scale_figures(self):
        pass

    def construct_graph(self, do_cliques=True, keep_mutual_only=True):
        self.compute_k_neighbors(keep_mutual_only=keep_mutual_only)
        edges = np.empty((0, 2), dtype=int)
        for i in range(self.figures_count):
            neighbors = list(self.k_neighbors[i])
            if do_cliques:
                new_edges = np.array(list(combinations(sorted(neighbors), 2)))
            else:
                new_edges = np.column_stack((np.full(len(neighbors), dtype=int, fill_value=i), neighbors))
            edges = np.vstack((edges, new_edges))
        self.graph_edges = np.unique(edges, axis=0)

    def compute_k_neighbors(self, keep_mutual_only=True):
        self.compute_dists()
        self.k_neighbors = []
        avg_k = 0
        for i in range(self.figures_count):
            self.k_neighbors.append(self.k_nearest_neighbors_for_fig(i))
        if not keep_mutual_only:
            self.avg_k = self.k
            return
        for i in range(self.figures_count):
            correct_neighbors = []
            for j in self.k_neighbors[i]:
                if i in self.k_neighbors[j]:
                    correct_neighbors.append(j)
                    avg_k += 1
            self.k_neighbors[i] = set(correct_neighbors)
        self.avg_k = avg_k / self.figures_count

    @abstractmethod
    def compute_dists(self):
        pass

    def normalize_dists(self):
        pass
        # dists_max = np.max(self.dists)
        # self.dists_scaler = dists_max
        # self.dists = self.dists / self.dists_scaler

    def normalize_new_distance(self, distance):
        return distance
        # return distance / self.dists_scaler

    def k_nearest_neighbors_for_fig(self, figure_idx) -> set[int]:
        distances = self.dists[figure_idx]
        neighbours = np.argsort(distances, kind='stable')[:self.k + 1]
        return set(neighbours)

    def solve_semidefinite_programming(self, max_iters=100):
        n = self.figures_count
        q = cvxpy.Variable((n, n), PSD=True)
        constraints = [cvxpy.trace(np.ones((n, n)) @ q) == 0]
        constraints += [cvxpy.trace(q) <= (n - 1) * self.c]

        constraints += [
            q[i][i] + q[j][j] - 2 * q[i][j] == self.dists[i, j] for i, j in
            self.graph_edges
        ]

        prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.trace(self.params @ self.params.T @ q)), constraints)

        prob.solve(solver=cvxpy.SCS, verbose=True, max_iters=max_iters, eps=1e-6)
        self.prob = prob
        # prob.solve()
        self.q = q.value

    def reduce_dimensions(self):
        eigenvectors, sigma = auxiliary.get_eigen_decomposition(self.q)
        self.embedded_y = np.dot(eigenvectors[:, :self.params_dim], np.sqrt(sigma[:self.params_dim, :self.params_dim]))

    def embedded_y_to_param_regression(self):
        lambda_ = 0.1
        I = np.eye(self.params.shape[1])
        self.B = np.linalg.solve(self.params.T @ self.params + lambda_ * I, self.params.T @ self.embedded_y)

    def compute_rre_median_embedding_vs_params(self):
        embedded_y_as_params = self.embedded_y_as_params()
        return np.median(
            np.linalg.norm(embedded_y_as_params - self.original_params, axis=1) / np.linalg.norm(embedded_y_as_params,
                                                                                                 axis=1))

    def embedded_y_as_params(self):
        standardized_y = np.dot(self.embedded_y, np.linalg.inv(self.B))
        return auxiliary.undo_standardize(standardized_y, means=self.params_means,
                                          stds=self.params_stds)

    def predict(self, figure, k=None, gd=False, plot_loss=False, baseline=False, symmetric=False):
        neighbors_cnt = k
        if k is None:
            neighbors_cnt = self.k
        if gd:
            prediction, loss = self.predictive_optimization_gd(figure)
        else:
            prediction, loss = self.predictive_optimization(y_nom=figure, plot_loss=plot_loss, k=neighbors_cnt,
                                                            baseline=baseline, symmetric=symmetric)
        prediction = auxiliary.undo_standardize(prediction, means=self.params_means, stds=self.params_stds)
        return prediction, loss

    def predictive_optimization(self, y_nom, k, plot_loss, baseline, seed=-1, symmetric=False):
        t0 = time.time()
        y_nom = self.center_and_scale_figure(y_nom)
        neighbours, distances = self.k_nearest_neighbours(y_nom, k, symmetric)
        t1 = time.time()

        lw = [-1.8] * self.params_dim
        up = [1.8] * self.params_dim

        def loss(x):
            return self.x_error(x, self.embedded_y[neighbours], distances)

        if baseline:
            x = self.params[neighbours[0]]
            return x, loss(x)

        if plot_loss:
            h = 100
            grid_points = np.array([np.linspace(lw[i], up[i], h) for i in range(len(lw))])
            losses = np.full(shape=(h, h), fill_value=0.1)
            for i in range(h):
                for j in range(h):
                    losses[j, i] = loss((grid_points[0, i], grid_points[1, j]))

            def plot_heatmap(losses, grid_points):
                plt.figure(figsize=(8, 6))
                x0, y0 = auxiliary.undo_standardize(grid_points[:, 0], means=self.params_means, stds=self.params_stds)
                x1, y1 = auxiliary.undo_standardize(grid_points[:, -1], means=self.params_means, stds=self.params_stds)
                print(x0, x1, y0, y1)
                plt.imshow(losses, extent=(x0, x1, y0, y1), origin='lower', interpolation='none', aspect='auto')
                plt.colorbar(label='Loss')
                plt.title('Heatmap of Losses')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.show()

            plot_heatmap(losses, grid_points)

        if seed == -1:
            x_opt = dual_annealing(loss, bounds=list(zip(lw, up)), maxfun=1000, maxiter=1)
        else:
            x_opt = dual_annealing(loss, bounds=list(zip(lw, up)), seed=seed, maxfun=10000, maxiter=3)
        t2 = time.time()
        print(f"Finding neighbors: {int((t1 - t0) * 1000)} ms , optimization: {int((t2 - t1) * 1000)} ms")
        return x_opt.x, loss(x_opt.x)

    def y_error(self, v, embedded_y_neighbors, distances):
        err_diff = (np.linalg.norm(v - embedded_y_neighbors, axis=1) -
                    distances)
        sum_err = np.sum(err_diff ** 2)
        return sum_err

    def x_error(self, x, embedded_y_neighbors, y_neighbors):
        return self.y_error(np.dot(x, self.B), embedded_y_neighbors, y_neighbors)

    @abstractmethod
    def center_and_scale_figure(self, figure):
        pass

    def predictive_optimization_gd(self, y_nom):
        t0 = time.time()
        y_nom = self.center_and_scale_figure(y_nom)
        neighbours, distances = self.k_nearest_neighbours(y_nom, k)
        t1 = time.time()

        lw = [-1.8] * self.params_dim
        up = [1.8] * self.params_dim

        weights = 1 / distances[neighbours] ** 2
        start_point = np.average(self.embedded_y[neighbours], axis=0, weights=weights)

        def loss(x):
            return self.x_error(x, self.embedded_y[neighbours], distances)

        res = minimize(loss, start_point, method='L-BFGS-B', bounds=list(zip(lw, up)))
        t2 = time.time()

        print(f"Finding neighbors: {int((t1 - t0) * 1000)} ms , optimization: {int((t2 - t1) * 1000)} ms")
        return res.x, loss(res.x)
