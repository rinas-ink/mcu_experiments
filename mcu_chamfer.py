import time

from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import dual_annealing, minimize
from sklearn.cluster import KMeans

import auxiliary
import numpy as np
import cvxpy
from itertools import combinations

from mcu_base import MCUbase


class MCUChamferModel(MCUbase):

    def __init__(self, k, c, figures, params, params_names=None):
        super().__init__(k, c, figures, params, params_names)
        self.sym_chamfer = None

    def center_and_scale_figures(self):
        max_distances = np.zeros(self.figures_count)
        self.figures = self.original_figures.copy()
        for i in range(self.figures_count):
            centroid = np.mean(self.original_figures[i], axis=0)
            self.figures[i] -= centroid
            max_distances[i] = np.max(np.linalg.norm(self.figures[i], axis=1))
        self.figures_scaler = np.max(max_distances)
        for i in range(len(self.figures)):
            self.figures[i] /= self.figures_scaler

    def center_and_scale_figure(self, figure):
        return figure / self.figures_scaler
        # return (figure - np.mean(figure, axis=0)) / self.figures_scaler

    def prepare_data(self):
        if self.params is not None:
            return
        self.params, self.params_means, self.params_stds = auxiliary.standardize(self.original_params)
        self.center_and_scale_figures()
        self.KD_figures_trees = [KDTree(figure.reshape(-1, 3)) for figure in self.figures]
        self.compute_sym_chamfer()

    def compute_sym_chamfer(self):
        if self.sym_chamfer is not None:
            return
        self.sym_chamfer = np.full(shape=(self.figures_count, self.figures_count, 3), fill_value=-1.0)
        for i in range(self.figures_count):
            self.sym_chamfer[i, i] = [0, 0, 0]
        # self.sym_chamfer = np.full(shape=(self.figures_count, self.figures_count), fill_value=-1.0)
        # np.fill_diagonal(self.sym_chamfer, 0)
        for i in range(self.figures_count):
            for j in range(i + 1, self.figures_count):
                points1 = self.figures[i]
                tree1 = self.KD_figures_trees[i]
                points2 = self.figures[j]
                tree2 = self.KD_figures_trees[j]
                dist1 = auxiliary.asymmetric_chamfer_dist(tree2, points1)
                dist2 = auxiliary.asymmetric_chamfer_dist(tree1, points2)
                if dist1 > dist2:
                    dist1, dist2 = dist2, dist1
                self.sym_chamfer[i, j] = np.array([(dist1 + dist2) / 2, dist1, dist2])
                # self.sym_chamfer[i, j] = auxiliary.symmetric_chamfer_dist(points1, tree1, points2, tree2)
                self.sym_chamfer[j, i] = self.sym_chamfer[i, j]

    def k_nearest_neighbours_sym_chamfer(self, figure_idx):
        distances = self.sym_chamfer[figure_idx][:, 0]
        neighbours = np.argsort(distances, kind='stable')[:self.k + 1]
        return neighbours

    def k_nearest_neighbours(self, points, k=None):
        neighbor_cnt = k
        if k is None:
            neighbor_cnt = self.k
        distances = []
        for i in range(self.figures_count):
            tree = self.KD_figures_trees[i]
            distances.append(auxiliary.asymmetric_chamfer_dist(tree, points))
        distances = np.array(distances)
        neighbours = np.argsort(distances, kind='stable')[:neighbor_cnt]
        return neighbours, distances[neighbours]

    # def k_nearest_neighbours(self, points, k=None):
    #     neighbor_cnt = k
    #     if k is None:
    #         neighbor_cnt = self.k
    #     distances = []
    #     points1 = points.reshape(-1, 3)
    #     tree1 = KDTree(points1)
    #     for i in range(self.figures_count):
    #         tree2 = self.KD_figures_trees[i]
    #         points2 = self.figures[i]
    #         distances.append(auxiliary.symmetric_chamfer_dist(points1, tree1, points2, tree2))
    #     distances = np.array(distances)
    #     neighbours = np.argsort(distances, kind='stable')[:neighbor_cnt]
    #     return neighbours, distances[neighbours]

    def construct_graph(self):
        if self.graph_edges is not None:
            return
        edges = np.empty((0, 2), dtype=int)
        for i in range(self.figures_count):
            neighbours = self.k_nearest_neighbours_sym_chamfer(i)
            all_pairs = np.array(list(combinations(sorted(neighbours), 2)))
            edges = np.vstack((edges, all_pairs))
        self.graph_edges = np.unique(edges, axis=0)

    def solve_semidefinite_programming(self, max_iters = 100):
        n = self.figures_count
        q = cvxpy.Variable((n, n), PSD=True)
        constraints = [cvxpy.trace(np.ones((n, n)) @ q) == 0]
        constraints += [cvxpy.trace(q) <= (n - 1) * self.c]

        constraints += [
            q[i][i] + q[j][j] - 2 * q[i][j] == self.sym_chamfer[i, j][0] for i, j in
            self.graph_edges
        ]

        prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.trace(self.params @ self.params.T @ q)), constraints)
        prob.solve(solver=cvxpy.SCS, verbose=False, max_iters=max_iters)
        # prob.solve()
        self.q = q.value

    def predictive_optimization_gd(self, y_nom):
        t0 = time.time()
        y_nom = (y_nom - np.mean(y_nom, axis=0)) / self.figures_scaler
        neighbours, distances = self.k_nearest_neighbors_sym_chamfer_new_fig(
            y_nom, k=4)
        t1 = time.time()

        weights = 1 / abs(distances)
        start_point = np.average(self.embedded_y[neighbours], axis=0, weights=weights)

        lw = [-1.3] * np.shape(self.B)[1]
        up = [1.3] * np.shape(self.B)[1]

        def loss(x):
            return self.x_error(x, y_nom, self.embedded_y[neighbours], distances)

        res = minimize(loss, start_point, method='L-BFGS-B', bounds=list(zip(lw, up)))
        t2 = time.time()

        print(f"Finding neighbors: {int((t1 - t0) * 1000)} ms , optimization: {int((t2 - t1) * 1000)} ms")
        return res.x, loss(res.x)


def diff_of_edges_lengths(ld_embedding, reconstructed_y, edges):
    edge_lengths_ld = np.linalg.norm(ld_embedding[edges[:, 0]] - ld_embedding[edges[:, 1]], axis=1)
    edge_lengths_rec = np.linalg.norm(reconstructed_y[edges[:, 0]] - reconstructed_y[edges[:, 1]], axis=1)
    return edge_lengths_ld - edge_lengths_rec
