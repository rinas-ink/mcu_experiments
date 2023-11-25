import random

import numpy as np
import matplotlib.pyplot as plt

N = 200


def get_p():
    return 2


def get_control_vars(size=N):
    return [[random.random() * 10 for _ in range(get_p())] for _ in range(size)]


def generate_random_cylinder_points(height, radius, num_points=100):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    heights = np.random.uniform(0, height, num_points)

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    points = np.column_stack((x, y, heights))

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


def generate_array_of_cylinders(control_vars, noise_level=0.1, num=N, num_points=100):
    cylinders = []
    for height, radius in control_vars:
        cylinder = generate_random_cylinder_points(height, radius, num_points)
        cylinder = add_noise_to_points(cylinder)
        cylinder = get_flat_array_of_cylinder(cylinder)
        cylinders.append(cylinder)
    return np.array(cylinders)


def main():
    cylinder = generate_random_cylinder_points(20, 5, 100)
    cylinder = add_noise_to_points(cylinder)
    plot_points(cylinder)


if __name__ == '__main__':
    main()
