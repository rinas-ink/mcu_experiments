import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    return {'xy': create_plane(point, scale, 'xy'),
            'yz': create_plane(point, scale, 'yz'),
            'xz': create_plane(point, scale, 'xz')}


def plot_plane(ax, x, y, z, color):
    ax.plot_surface(x, y, z, alpha=0.5, rstride=1, cstride=1, color=color)


def plot_figure(ax, planes):
    plot_plane(ax, *planes['xy'], color='g')
    plot_plane(ax, *planes['yz'], color='b')
    plot_plane(ax, *planes['xz'], color='r')


def get_control_vars(n=3):
    return [random.randint(5, 15) for _ in range(n)]

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    initial_point = (0, 0, 0)
    scale = get_control_vars(10)

    planes = create_figure(initial_point, scale)
    plot_figure(ax, planes)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim([-1, initial_point[0] + scale[0] + 1])
    ax.set_ylim([-1, initial_point[1] + scale[1] + 1])
    ax.set_zlim([-1, initial_point[2] + scale[2] + 1])

    plt.show()


if __name__ == '__main__':
    main()
