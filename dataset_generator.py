import math
import numpy as np


def get_control_vars(deterministic=True, dimensionality=2, size=100, lw=None, up=None):
    if lw is None:
        lw = [1] * dimensionality
    if up is None:
        up = [10] * dimensionality
    if deterministic:
        grid_size = int(math.ceil(math.pow(size, 1 / dimensionality)))
        grid_vars = [np.linspace(lw[i], up[i], grid_size) for i in range(dimensionality)]
        grids = np.meshgrid(*grid_vars)
        return np.vstack([grid.ravel() for grid in grids]).T
    return np.array([[np.random.random() * (up[i] - lw[i]) + lw[i] for i in range(dimensionality)] for _ in range(size)])


def add_noise_to_points(points, noise_level=0.1):
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise
    return noisy_points
