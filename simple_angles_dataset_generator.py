import math

import auxiliary
import dataset_generator

import numpy as np


def generate_angles_points(min_num_points, inside_angle=90, rotation_angle_x=0, rotation_angle_y=0,
                           rotation_angle_z=0):
    """
    Generates angle, that looks loke folded by one side piece of paper
    :param min_num_points:
    :param inside_angle: angle at the edge of figure, in degrees
    :param rotation_angle_x: rotation angle along X axis, degrees
    :param rotation_angle_y: rotation angle along Y axis, degrees
    :param rotation_angle_z: rotation angle along Z axis, degrees
    """
    sqrt_num_points = int(math.ceil(math.sqrt(min_num_points / 2)))
    length = 1
    width = 1
    length_coords = np.linspace(0, length, sqrt_num_points)
    width_coords1 = np.linspace(0, width, sqrt_num_points)
    X1, Y1 = np.meshgrid(length_coords, width_coords1)
    first_plane_points = np.vstack((X1.ravel(), Y1.ravel(), np.zeros_like(X1).ravel())).T
    width_coords2 = np.linspace(1 / sqrt_num_points, width, sqrt_num_points)
    X2, Y2 = np.meshgrid(length_coords, width_coords2)
    second_plane_points = np.vstack((X2.ravel(), Y2.ravel(), np.zeros_like(X2).ravel())).T
    second_plane_points = dataset_generator.rotate_points(second_plane_points, angle_x=inside_angle)
    angle_points = np.concatenate((first_plane_points, second_plane_points))
    angle_points = angle_points - np.mean(angle_points, axis=0)
    angle_points = dataset_generator.rotate_points(angle_points, rotation_angle_x, rotation_angle_y, rotation_angle_z)
    return angle_points


def main():
    angle1 = generate_angles_points(100, 90, 0)
    angle1 = dataset_generator.add_noise_to_points(angle1, 0.01)
    angle1, mean = auxiliary.center(angle1)
    dataset_generator.plot_points(angle1)
    angle2 = generate_angles_points(100, 45, 0)
    angle2 = dataset_generator.add_noise_to_points(angle2, 0.01)
    angle2, mean = auxiliary.center(angle1)


if __name__ == '__main__':
    main()
