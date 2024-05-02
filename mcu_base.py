import time
from abc import abstractmethod

from matplotlib import pyplot as plt
from scipy.optimize import dual_annealing, minimize

import auxiliary
import numpy as np
import cvxpy
from itertools import combinations


class MCUbase:
    def __init__(self, k, c, figures, params, params_names=None, M=None):
        # TODO remove params dimensions that are almost constant
        """
        :param k:
        :param c:
        :param figures:  list[np.array(point count, 3)]
        :param params:
        :param params_names:
        """
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
        self.M = M
        if M is None:
            self.M = self.params_dim

    def train(self, max_iters = 100):
        self.prepare_data()
        self.construct_graph()
        self.solve_semidefinite_programming(max_iters)
        self.reduce_dimensions()
        self.embedded_y_to_param_regression()

    def predict(self, figure, gd=False, plot_loss=False):
        if gd:
            prediction, loss = self.predictive_optimization_gd(figure)
        else:
            prediction, loss = self.predictive_optimization(y_nom=figure, plot_loss=plot_loss)
        prediction = auxiliary.undo_standardize(prediction, means=self.params_means, stds=self.params_stds)
        return prediction, loss

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def k_nearest_neighbours(self, y, k=None):
        pass

    @abstractmethod
    def construct_graph(self):
        pass

    @abstractmethod
    def solve_semidefinite_programming(self, max_iters):
        pass

    @abstractmethod
    def center_and_scale_figure(self, figure):
        pass

    def reduce_dimensions(self):
        eigenvectors, sigma = auxiliary.get_eigen_decomposition(self.q)
        self.embedded_y = np.dot(eigenvectors[:, :self.M], np.sqrt(sigma[:self.M, :self.M]))

    def embedded_y_to_param_regression(self):
        lambda_ = 0.1
        I = np.eye(self.params.shape[1])
        self.B = np.linalg.solve(self.params.T @ self.params + lambda_ * I, self.params.T @ self.embedded_y)

    def embedded_y_as_params(self):
        standardized_y = np.dot(self.embedded_y, np.linalg.inv(self.B))
        return auxiliary.undo_standardize(standardized_y, means=self.params_means,
                                          stds=self.params_stds)

    def compute_rre_median_embedding_vs_params(self):
        embedded_y_as_params = self.embedded_y_as_params()
        return np.median(
            np.linalg.norm(embedded_y_as_params - self.original_params, axis=1) / np.linalg.norm(embedded_y_as_params,
                                                                                                 axis=1))

    def y_error(self, v, embedded_y_neighbors, distances):
        err_diff = (np.linalg.norm(v - embedded_y_neighbors, axis=1) -
                    distances)
        sum_err = np.sum(err_diff ** 2)
        return sum_err

    def x_error(self, x, embedded_y_neighbors, y_neighbors):
        return self.y_error(np.dot(x, self.B), embedded_y_neighbors, y_neighbors)

    def predictive_optimization(self, y_nom, plot_loss, seed=-1):
        t0 = time.time()
        y_nom = self.center_and_scale_figure(y_nom)
        neighbours, distances = self.k_nearest_neighbours(y_nom)
        t1 = time.time()

        lw = [-1.8] * self.params_dim
        up = [1.8] * self.params_dim

        def loss(x):
            return self.x_error(x, self.embedded_y[neighbours], distances)

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
            x_opt = dual_annealing(loss, bounds=list(zip(lw, up)), seed=seed, maxfun=1000, maxiter=3)
        t2 = time.time()
        print(f"Finding neighbors: {int((t1 - t0) * 1000)} ms , optimization: {int((t2 - t1) * 1000)} ms")
        return x_opt.x, loss(x_opt.x)

    def predictive_optimization_gd(self, y_nom):
        return 0, 0
