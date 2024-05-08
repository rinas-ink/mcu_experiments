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
        self.original_figures = np.array(self.original_figures).reshape(self.figures_count, -1)

    def center_and_scale_figures(self):
        means = np.mean(self.original_figures, axis=0)
        self.figures, self.figures_means = self.original_figures - means, means
        self.figures_scaler = np.average(np.std(self.figures, axis=0))
        self.figures = self.figures / self.figures_scaler

    def compute_dists(self):
        if self.dists is not None:
            return
        self.dists = np.full(shape=(self.figures_count, self.figures_count), fill_value=-1.0)
        np.fill_diagonal(self.dists, 0)
        for i in range(self.figures_count):
            for j in range(i + 1, self.figures_count):
                self.dists[i, j] = np.linalg.norm(self.figures[i] - self.figures[j]) / self.figures.shape[1]
                self.dists[j, i] = self.dists[i, j]
        self.normalize_dists()

    def center_and_scale_figure(self, figure):
        figure = figure.reshape(-1)
        return (figure - self.figures_means) / self.figures_scaler

    def k_nearest_neighbours(self, y, k=None, symmetric=False):
        neighbor_cnt = k
        if k is None:
            neighbor_cnt = self.k
        distances = np.linalg.norm(self.figures - y, axis=1) / self.figures.shape[1]
        distances = self.normalize_new_distance(distances)
        neighbours = np.argsort(distances, kind='stable')[:neighbor_cnt + 1]
        return neighbours, distances[neighbours]
