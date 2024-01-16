import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dataset_generator

N = 200


def get_p():
    return 2


def swiss_roll_function(x, y, c1, c2):
    r = 1 + c2 / 10
    x_val = (4 / 9 * c1 + 50 / 9) * x * np.cos(2 * np.pi * r * (x - 4) / 12)
    z_val = (4 / 9 * c1 + 50 / 9) * x * np.sin(2 * np.pi * r * (x - 4) / 12)
    return x_val, y, z_val


def get_meshgrid(n):
    """
    :return: n*n meshgrid
    """
    x = np.linspace(4, 16, n, endpoint=False)
    y = np.linspace(4, 16, n, endpoint=False)
    return np.meshgrid(x, y)


def add_noise(x, y, noise_strength=0.1):
    x += np.random.normal(0, noise_strength, x.shape)
    y += np.random.normal(0, noise_strength, y.shape)


def visualize_swiss_roll(x, y, z):
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=x, cmap=plt.cm.Spectral, s=4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def get_flat_array_of_swiss_roll(x_swiss_roll, y_swiss_roll, z_swiss_roll):
    return np.column_stack((x_swiss_roll, y_swiss_roll, z_swiss_roll)).reshape(-1)


def generate_array_of_swiss_rolls(control_vars, noise_level=0.1, min_num_points=1600):
    """
    :return: List N*M, where M is from the article = len (all points concatenated)
    """
    sqrt_num_points = int(math.ceil(math.sqrt(min_num_points)))
    X, Y = get_meshgrid(sqrt_num_points)
    X, Y = X.ravel(), Y.ravel()
    samples = []
    for i in range(len(control_vars)):
        c_size, c_degree = control_vars[i]
        x_swiss_roll, y_swiss_roll, z_swiss_roll = swiss_roll_function(X, Y, c_size, c_degree)
        # visualize_swiss_roll(x_swiss_roll, y_swiss_roll, z_swiss_roll)
        # break
        sample = get_flat_array_of_swiss_roll(x_swiss_roll, y_swiss_roll, z_swiss_roll)
        samples.append(sample)

    samples = dataset_generator.add_noise_to_points(np.array(samples), noise_level=noise_level)

    return samples
