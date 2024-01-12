import matplotlib.pyplot as plt
import numpy as np


def init_viz():
    fig = plt.figure()
    return fig.add_subplot(111, projection='3d')


def plot_plane(ax, x, y, z, color):
    ax.plot_surface(x, y, z, alpha=0.5, rstride=1, cstride=1, color=color)


def plot_figure(ax, planes):
    plot_plane(ax, *planes[0], color='g')
    plot_plane(ax, *planes[1], color='b')
    plot_plane(ax, *planes[2], color='r')


def set_limits(ax, initial_point, scale):
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim([initial_point, scale])
    ax.set_ylim([initial_point, scale])
    ax.set_zlim([initial_point, scale])


def change_view_point(ax):
    ax.view_init(elev=20., azim=30)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def show():
    plt.show()
