import math
import numpy as np
from matplotlib import pyplot as plt


def default_params_names(p):
    return np.array([f'param{i}' for i in range(p)])


def get_control_vars(deterministic=True, dimensionality=2, size=100, lw=None, up=None, seed=None):
    if seed is None:
        rand = np.random.default_rng()
    else:
        rand = np.random.default_rng(seed=seed)
    if lw is None:
        lw = [1] * dimensionality
    if up is None:
        up = [10] * dimensionality
    if deterministic:
        grid_size = int(math.ceil(math.pow(size, 1 / dimensionality)))
        grid_vars = [np.linspace(lw[i], up[i], grid_size) for i in range(dimensionality)]
        grids = np.meshgrid(*grid_vars)
        return np.vstack([grid.ravel() for grid in grids]).T
    return np.array(
        [[rand.random() * (up[i] - lw[i]) + lw[i] for i in range(dimensionality)] for _ in range(size)])


def put_control_vars_in_dict(control_vars, p, param_names):
    if p != len(param_names):
        raise 'Provide names for all parameters that match parameters of generator function'
    result = []
    for params in control_vars:
        param_dict = {param_names[j]: params[j] for j in range(p)}
        result.append(param_dict)
    return np.array(result)


def add_noise_to_points(points, noise_level=0.1):
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise
    return noisy_points


def translate_points(points, translation_vector):
    return points + translation_vector


def rotate_points(points, angle_x=0, angle_y=0, angle_z=0):
    """
    :param points: np array of tuples [x, y, z], that represent 3d coordinates of point
    :param angle_x: around X axis, in degrees
    :param angle_y: around Y axis, in degrees
    :param angle_z: around Z axis, in degrees
    :return: rotated cloud of points
    """
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)
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


def plot_points(points, elev=30, azim=60, limits = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, c=points[:, 0], cmap=plt.cm.Spectral)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if limits is not None:
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
    ax.view_init(elev=elev, azim=azim)  # set the elevation and azimuth angles
    plt.show()


def generate_array_of_figures(control_vars, figure_generator, noise_level=0.1, min_num_points=100, fixed_params=None):
    if fixed_params is None:
        fixed_params = {}
    figures = []
    for params in control_vars:
        points = figure_generator(min_num_points, **params, **fixed_params)
        points = add_noise_to_points(points, noise_level)
        figures.append(np.array(points))
    return figures


def add_translations_to_figures(figures, translation_cnt):
    """
    Returns new dataset, that is translation_cnt^3 times bigger then original.

    """
