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
        self.KD_figures_trees = None

    def prepare_data(self):
        super().prepare_data()
        self.KD_figures_trees = [KDTree(figure.reshape(-1, 3)) for figure in self.figures]

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

    def compute_dists(self):
        if self.dists is not None:
            return
        # self.dists = np.full(shape=(self.figures_count, self.figures_count, 3), fill_value=-1.0)
        # for i in range(self.figures_count):
        #     self.dists[i, i] = [0, 0, 0]
        self.dists = np.full(shape=(self.figures_count, self.figures_count), fill_value=-1.0)
        np.fill_diagonal(self.dists, 0)
        for i in range(self.figures_count):
            for j in range(i + 1, self.figures_count):
                points1 = self.figures[i]
                tree1 = self.KD_figures_trees[i]
                points2 = self.figures[j]
                tree2 = self.KD_figures_trees[j]
                # dist1 = auxiliary.asymmetric_chamfer_dist(tree2, points1)
                # dist2 = auxiliary.asymmetric_chamfer_dist(tree1, points2)
                # if dist1 > dist2:
                #     dist1, dist2 = dist2, dist1
                # self.dists[i, j] = np.array([(dist1 + dist2) / 2, dist1, dist2])
                self.dists[i, j] = auxiliary.symmetric_chamfer_dist(points1, tree1, points2, tree2)
                self.dists[j, i] = self.dists[i, j]

    def center_and_scale_figure(self, figure):
        return figure / self.figures_scaler
        # return (figure - np.mean(figure, axis=0)) / self.figures_scaler

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
