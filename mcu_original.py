import time

import cvxpy
from scipy.optimize import dual_annealing, minimize

import auxiliary
import numpy as np
from itertools import combinations

from mcu_base import MCUbase


class MCUOriginalModel(MCUbase):

    def __init__(self, k, c, figures, params, params_names=None):
        super().__init__(k, c, figures, params, params_names)
        self.dists = None
        self.original_figures = np.array(self.original_figures).reshape(self.figures_count, -1)

    def compute_dists(self):
        if self.dists is not None:
            return
        self.dists = np.full(shape=(self.figures_count, self.figures_count), fill_value=-1.0)
        np.fill_diagonal(self.dists, 0)
        for i in range(self.figures_count):
            for j in range(i + 1, self.figures_count):
                self.dists[i, j] = np.linalg.norm(self.figures[i] - self.figures[j])
                self.dists[j, i] = self.dists[i, j]

    def center_and_scale_figures(self):
        means = np.mean(self.original_figures, axis=0)
        self.figures, self.figures_means = self.original_figures - means, means
        self.figures_scaler = np.average(np.std(self.figures, axis=0))
        self.figures = self.figures / self.figures_scaler

    def center_and_scale_figure(self, figure):
        figure = figure.reshape(-1)
        return (figure - self.figures_means) / self.figures_scaler

    def prepare_data(self):
        self.params, self.params_means, self.params_stds = auxiliary.standardize(self.original_params)
        self.center_and_scale_figures()

    def k_nearest_neighbours(self, y, k=None):
        neighbor_cnt = k
        if k is None:
            neighbor_cnt = self.k
        distances = np.round(np.linalg.norm(self.figures - y, axis=1), 12)
        neighbours = np.argsort(distances, kind='stable')[:neighbor_cnt + 1]
        return neighbours, distances[neighbours]

    def construct_graph(self):
        edges = np.empty((0, 2), dtype=int)
        for y in self.figures:
            neighbours, _ = self.k_nearest_neighbours(y)
            all_pairs = np.array(list(combinations(sorted(neighbours), 2)))
            edges = np.vstack((edges, all_pairs))
        self.graph_edges = np.unique(edges, axis=0)

    def solve_semidefinite_programming(self, max_iters = 100):
        n = self.figures_count
        p = np.dot(self.figures, self.figures.T)
        q = cvxpy.Variable((n, n), PSD=True)
        constraints = [cvxpy.trace(np.ones((n, n)) @ q) == 0]
        constraints += [cvxpy.trace(q) <= (n - 1) * self.c]
        constraints += [
            q[i][i] + q[j][j] - 2 * q[i][j] == p[i][i] + p[j][j] - 2 * p[i][j] for i, j in self.graph_edges
        ]

        prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.trace(self.params @ self.params.T @ q)), constraints)
        prob.solve(solver=cvxpy.SCS, max_iters = max_iters)
        # prob.solve()
        self.q = q.value

    def predictive_optimization_gd(self, y_nom):
        t0 = time.time()
        y_nom = (y_nom - self.figures_means) / self.figures_scaler
        distances = np.linalg.norm(self.figures - y_nom, axis=1)
        neighbours = np.argsort(distances)[:self.k]
        t1 = time.time()

        weights = 1 / distances[neighbours] ** 2
        start_point = np.average(self.embedded_y[neighbours], axis=0, weights=weights)

        lw = [-1.3] * np.shape(self.B)[1]
        up = [1.3] * np.shape(self.B)[1]

        def loss(x):
            return self.x_error(x, y_nom, self.embedded_y[neighbours], self.figures[neighbours])

        res = minimize(loss, start_point, method='L-BFGS-B', bounds=list(zip(lw, up)))
        t2 = time.time()

        print(f"Finding neighbors: {int((t1 - t0) * 1000)} ms , optimization: {int((t2 - t1) * 1000)} ms")
        return res.x, loss(res.x)
