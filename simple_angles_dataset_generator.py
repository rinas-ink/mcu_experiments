import math
import dataset_generator

import numpy as np


def generate_angles_points(min_num_points, inside_angle, rotation_angle_x=0, rotation_angle_y=0, rotation_angle_z=0):
    sqrt_num_points = int(math.ceil(math.sqrt(min_num_points / 2)))
    side_coords1 = np.linspace(0, 1, sqrt_num_points)
    X1, Y1 = np.meshgrid(side_coords1, side_coords1)
    first_plane_points = np.vstack((X1.ravel(), Y1.ravel(), np.zeros_like(X1).ravel())).T
    side_coords2 = np.linspace(1 / sqrt_num_points, 1, sqrt_num_points)
    X2, Y2 = np.meshgrid(side_coords1, side_coords2)
    second_plane_points = np.vstack((X2.ravel(), Y2.ravel(), np.zeros_like(X2).ravel())).T
    second_plane_points = dataset_generator.rotate_points(second_plane_points, angle_x=inside_angle)
    angle_points = np.concatenate((first_plane_points, second_plane_points))
    angle_points = dataset_generator.rotate_points(angle_points, rotation_angle_x, rotation_angle_y, rotation_angle_z)
    return angle_points


def main():
    angle = generate_angles_points(100, 45, 0)
    dataset_generator.plot_points(angle)


if __name__ == '__main__':
    main()
