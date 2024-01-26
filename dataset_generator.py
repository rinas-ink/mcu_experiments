import math
import random

import numpy as np
from matplotlib import pyplot as plt


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
    return np.array([[random.random() * (up[i] - lw[i]) + lw[i] for i in range(dimensionality)] for _ in range(size)])


def add_noise_to_points(points, noise_level=0.1):
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise
    return noisy_points


def rotate_points(points, angle_x=0, angle_y=0, angle_z=0):
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    rotation_matrix_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])

    rotation_matrix_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]])

    points = np.dot(points, rotation_matrix_x.T)
    points = np.dot(points, rotation_matrix_y.T)
    points = np.dot(points, rotation_matrix_z.T)
    return points


def plot_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, c=points[:, 0], cmap=plt.cm.Spectral)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def generate_array_of_figures(control_vars, figure_generator, noise_level=0.1, min_num_points=100):
    figures = []
    for params in control_vars:
        points = figure_generator(min_num_points, *params)
        points = add_noise_to_points(points, noise_level)
        points = points.reshape(-1)
        figures.append(points)
    return np.array(figures)
