import math
import dataset_generator

import numpy as np
import matplotlib.pyplot as plt

N = 200


def get_p():
    return 2


def generate_cylinder_points(height, radius, sorted, min_num_points=100, deterministic_scatter=False):
    """
    :param min_num_points: if deterministic_scatter=True then there generated a number of points,
     equal to the first square >= min_num_points
    """
    angles = None
    heights = None
    sqrt_num_points = int(math.ceil(math.sqrt(min_num_points)))
    min_num_points = sqrt_num_points ** 2
    if not deterministic_scatter:
        angles = np.random.uniform(0, 2 * np.pi, min_num_points)
        heights = np.random.uniform(0, height, min_num_points)
    else:
        angles = np.linspace(0, 2 * np.pi, sqrt_num_points, endpoint=False)
        angles = np.repeat(angles, sqrt_num_points)
        heights = np.linspace(0, height, sqrt_num_points)
        heights = np.repeat(heights, sqrt_num_points)

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    points = np.column_stack((x, y, heights))

    if sorted:
        sorted_indices = np.lexsort((points[:, 0], points[:, 1], points[:, 2]))
        return points[sorted_indices]
    else:
        np.random.shuffle(points)
    return points


def generate_rectangle(width, height, sorted, num_points=100, deterministic_scatter=False):
    sqrt_num_points = int(math.ceil(math.sqrt(num_points)))
    y = None
    x = None
    if not deterministic_scatter:
        y = np.random.uniform(0, height, sqrt_num_points)
        x = np.random.uniform(0, width, sqrt_num_points)
    else:
        y = np.linspace(0, height, sqrt_num_points)
        x = np.linspace(0, width, sqrt_num_points)

    points = np.meshgrid(x, y)

    if sorted:
        sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
        return points[sorted_indices]
    return np.random.shuffle(np.array(points))


def plot_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()


def get_flat_array_of_cylinder(points):
    return np.column_stack(points).reshape(-1)


def generate_array_of_cylinders(control_vars, noise_level=0.1, num_points=100, sorted=False,
                                deterministic_scatter=False):
    cylinders = []
    for height, radius in control_vars:
        cylinder = generate_cylinder_points(height, radius, sorted, num_points, deterministic_scatter)
        cylinder = dataset_generator.add_noise_to_points(cylinder, noise_level)
        cylinder = get_flat_array_of_cylinder(cylinder)
        cylinders.append(cylinder)
    return np.array(cylinders)


def main():
    cylinder = generate_cylinder_points(20, 5, False, 100)
    cylinder = dataset_generator.add_noise_to_points(cylinder)
    plot_points(cylinder)


if __name__ == '__main__':
    main()
