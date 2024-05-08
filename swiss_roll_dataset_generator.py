import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dataset_generator

N = 200


def get_p():
    return 2


def generate_swiss_roll(min_num_points, c1, c2):
    sqrt_num_points = int(math.ceil(math.sqrt(min_num_points)))
    X, Y = get_meshgrid(sqrt_num_points)
    X, Y = X.ravel(), Y.ravel()
    r = 1 + c2 / 10
    x_val = (4 / 9 * c1 + 50 / 9) * X * np.cos(2 * np.pi * r * (X - 4) / 12)
    z_val = (4 / 9 * c1 + 50 / 9) * X * np.sin(2 * np.pi * r * (X - 4) / 12)
    points = np.column_stack((x_val, Y, z_val))
    points = points - np.mean(points, axis=0)
    # new_points = []
    # for i in range(points.shape[0]):
    #     if points[i, 1] > 0 and points[i, 1] >0:
    #         new_points.append(points[i, :])
    # points = np.array(new_points)
    return points


def get_meshgrid(n):
    """
    :return: n*n meshgrid
    """
    x = np.linspace(4, 16, n, endpoint=False)
    y = np.linspace(4, 16, n, endpoint=False)
    return np.meshgrid(x, y)


def add_noise(x, noise_strength=0.1):
    x += np.random.normal(0, noise_strength, x.shape)


def visualize_swiss_roll(x, y, z):
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=x, cmap=plt.cm.Spectral, s=4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


