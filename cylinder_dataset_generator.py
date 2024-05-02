import math
import dataset_generator
import itertools

import numpy as np


def generate_cylinder_points(min_num_points, height=50, radius=5, rotation_angle_x=0, rotation_angle_y=0):
    """
    Rotation around other axis then x doesn't make sense because cylinder is symmetric
    :param min_num_points: if deterministic_scatter=True then there generated a number of points,
     equal to the first square >= min_num_points
    """
    sqrt_num_points = int(math.ceil(math.sqrt(min_num_points)))

    heights = np.linspace(0, height, sqrt_num_points)
    angles = np.linspace(0, 2 * np.pi, sqrt_num_points, endpoint=False)
    angles = np.vstack((radius * np.cos(angles), radius * np.sin(angles))).T

    combinations = list(itertools.product(angles, heights))
    points = np.array([[angle[0], angle[1], height] for angle, height in combinations])
    points = points - np.mean(points, axis=0)
    # new_points = []
    # for i in range(points.shape[0]):
    #     if points[i, 1] > 0 and points[i, 1] >0:
    #         new_points.append(points[i, :])
    # points = np.array(new_points)
    points = dataset_generator.rotate_points(points, angle_x=rotation_angle_x, angle_y=rotation_angle_y)

    return points


def main():
    cylinder = generate_cylinder_points(300, 10, 3.5)
    cylinder = dataset_generator.add_noise_to_points(cylinder)
    dataset_generator.plot_points(cylinder)


if __name__ == '__main__':
    main()
