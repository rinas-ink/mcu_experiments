import math
import dataset_generator
import itertools

import numpy as np


def generate_cylinder_points(min_num_points, height, radius):
    """
    :param min_num_points: if deterministic_scatter=True then there generated a number of points,
     equal to the first square >= min_num_points
    """
    sqrt_num_points = int(math.ceil(math.sqrt(min_num_points)))

    heights = np.linspace(0, height, sqrt_num_points)
    angles = np.linspace(0, 2 * np.pi, sqrt_num_points, endpoint=False)
    angles = np.vstack((radius * np.cos(angles), radius * np.sin(angles))).T

    combinations = list(itertools.product(angles, heights))
    points = [[angle[0], angle[1], height] for angle, height in combinations]

    return np.array(points)

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    points = np.column_stack((x, y, heights))

    return points


def main():
    cylinder = generate_cylinder_points(100, 20, 5)
    cylinder = dataset_generator.add_noise_to_points(cylinder)
    dataset_generator.plot_points(cylinder)


if __name__ == '__main__':
    main()
