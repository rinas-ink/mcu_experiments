import numpy as np

import dataset_generator
from vizualize_angles import plot_figure, init_viz, show, set_labels, set_title

N = 50


def get_p():
    return 3


def get_min_value():
    return 1


def get_max_value():
    return 50



def create_plane(point, scale, plane, num=20):
    axes = {'xy': (0, 1, 2), 'yz': (1, 2, 0), 'xz': (0, 2, 1)}
    a_idx, b_idx, c_idx = axes[plane]

    a = np.linspace(point[a_idx], point[a_idx] + scale[a_idx], num)
    b = np.linspace(point[b_idx], point[b_idx] + scale[b_idx], num)
    A, B = np.meshgrid(a, b)
    C = np.full_like(A, point[c_idx])

    planes = {'xy': (A, B, C), 'yz': (C, A, B), 'xz': (A, C, B)}
    return planes[plane]


def get_radians(angles):
    return map(np.radians, angles)


def rotate_planes(planes, rotation_matrix):
    rotated_planes = []
    for plane in planes:
        xy, yz, xz = plane
        rotated_points = np.dot(rotation_matrix, np.array([xy.ravel(), yz.ravel(), xz.ravel()]))
        xy_rotated, yz_rotated, xz_rotated = rotated_points.reshape(3, xy.shape[0], xy.shape[1])
        rotated_planes.append((xy_rotated, yz_rotated, xz_rotated))
    return rotated_planes


def create_figure(point, scale, noise=True, angles=(0, 0, 0)):
    planes = [create_plane(point, scale, 'xy', noise),
              create_plane(point, scale, 'yz', noise),
              create_plane(point, scale, 'xz', noise)]

    rotated_planes = [dataset_generator.rotate_points(plane, *angles) for plane in planes]

    return rotated_planes


def get_control_vars(n=get_p(), min_value=get_min_value(), max_value=get_max_value()):
    return np.array([np.random.randint(min_value, max_value) for _ in range(n)])


def get_array_of_control_vars(noise=True, dim=get_p(), size=N,
                              min_value=get_min_value(), max_value=get_max_value()):
    """
    :param noise: return random matrix if True
    :param dim: dimension of control variable (P from the article)
    :param size: amount of them (N from the article)
    :param min_value: minimal possible value
    :param max_value: maximum possible value
    :return: ndarray N*P
    """
    if noise:
        return [get_control_vars(dim, min_value, max_value) for _ in range(size)]
    else:
        matrix = np.zeros((size, dim))

        for i in range(size):
            for j in range(dim):
                matrix[i, j] = (i+j) % max_value + 1
        return matrix

def get_array_of_figures(control_vars, num=N, noise=True):
    samples = []
    for i in range(num):
        initial_point = (0, 0, 0)
        length, width, height = control_vars[i]
        xy, yz, xz = create_figure(initial_point, scale=(length, width, height), noise=noise)
        sample = dataset_generator.get_flat_array_of_points((xy, yz, xz))
        samples.append(sample)
    return samples


def viz_rotation():
    ax = init_viz()
    initial_point = (0, 0, 0)
    angels = (45, 60, 120)
    scale = get_control_vars(get_p())
    planes = create_figure(initial_point, scale, angles=angels)
    plot_figure(ax, planes)
    set_labels(ax)
    set_title(*angels)
    show()


if __name__ == '__main__':
    viz_rotation()
