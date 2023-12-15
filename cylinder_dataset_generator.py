import math
import random

import numpy as np
import matplotlib.pyplot as plt

N = 200


def get_p():
    return 2


def get_control_vars(size=N, lw=1, up=10):
    height = np.linspace(lw, up, int(math.sqrt(size)))
    width = np.linspace(lw, up, int(math.sqrt(size)))

    x_grid, y_grid = np.meshgrid(height, width)

    pairs = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    return pairs


def get_random_control_vars(size=N):
    return [[random.random() * 10 for _ in range(get_p())] for _ in range(size)]


def generate_cylinder_points(height, radius, sorted, num_points=100, random_scatter=False):
    angles = None
    heights = None
    if random:
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        heights = np.random.uniform(0, height, num_points)
    else:
        angles = np.linspace(0, 2 * np.pi, num_points)
        heights = np.linspace(0, height, num_points)

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    points = np.column_stack((x, y, heights))
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))

    if sorted:
        return points[sorted_indices]
    return points


def add_noise_to_points(points, noise_level=0.1):
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise
    return noisy_points


def plot_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()


def get_flat_array_of_cylinder(points):
    return np.column_stack(points).reshape(-1)


def generate_array_of_cylinders(control_vars, noise_level=0.1, num_points=100, sorted=False, random_scatter=False):
    cylinders = []
    for height, radius in control_vars:
        cylinder = generate_cylinder_points(height, radius, sorted, num_points, random_scatter)
        cylinder = add_noise_to_points(cylinder, noise_level)
        cylinder = get_flat_array_of_cylinder(cylinder)
        cylinders.append(cylinder)
    return np.array(cylinders)


def main():
    cylinder = generate_cylinder_points(20, 5, False, 100)
    cylinder = add_noise_to_points(cylinder)
    plot_points(cylinder)


if __name__ == '__main__':
    main()
