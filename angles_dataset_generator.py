import random
import numpy as np

from vizualize_angles import plot_figure, init_viz, show, set_limits, change_view_point

N = 50
def get_p():
    return 3


def get_max_value():
    return 50


def get_min_value():
    return 15


def add_noise(A, B, C, noise_level=0.05):
    A += np.random.normal(0, noise_level, A.shape)
    B += np.random.normal(0, noise_level, B.shape)
    C += np.random.normal(0, noise_level, C.shape)


def create_plane(point, scale, plane, num=20):
    axes = {'xy': (0, 1, 2), 'yz': (1, 2, 0), 'xz': (0, 2, 1)}
    a_idx, b_idx, c_idx = axes[plane]

    a = np.linspace(point[a_idx], point[a_idx] + scale[a_idx], num)
    b = np.linspace(point[b_idx], point[b_idx] + scale[b_idx], num)
    A, B = np.meshgrid(a, b)
    C = np.full_like(A, point[c_idx])

    add_noise(A, B, C)

    planes = {'xy': (A, B, C), 'yz': (C, A, B), 'xz': (A, C, B)}
    return planes[plane]


def create_figure(point, scale):
    return [create_plane(point, scale, 'xy'),
        create_plane(point, scale, 'yz'),
        create_plane(point, scale, 'xz')]


def main():
    ax = init_viz()
    initial_point = (0, 0, 0)
    scale = get_control_vars(get_p())
    planes = create_figure(initial_point, scale)
    plot_figure(ax, planes)
    set_limits(ax, 0, get_max_value())
    change_view_point(ax)

    show()


def get_control_vars(n=get_p(), size=N):
    """
    :param n: dimension of control variable (P from the article)
    :param size: amount of them
    :return: ndarray N*P
    """
    return np.array([[random.randint(1, 10) for _ in range(n)] for _ in range(size)])


def get_flat_array_of_3d_data(x, y, z):
    return np.column_stack((x, y, z)).reshape(-1)


def get_array_of_figures(control_vars, num=N):
    samples = []
    for i in range(num):
        initial_point = (0, 0, 0)
        length, width, height = control_vars[i]
        xy, yz, xz = create_figure(initial_point, scale=(length, width, height))
        sample = get_flat_array_of_3d_data(xy, yz, xz)
        samples.append(sample)
    return samples


if __name__ == '__main__':
    main()
