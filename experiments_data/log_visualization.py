import sys
import numpy as np
import matplotlib.pyplot as plt


def get_data_from_file(file_name):
    file = open(file_name, "r")
    lines = file.readlines()
    file.close()

    data = []
    for line in lines:
        splitted = line.split()
        if len(splitted) != 7 or splitted[0][-1] != '|':
            continue
        data.append([int(splitted[0][:-1]), float(splitted[1]), float(splitted[2])])

    return np.array(data)


def add_line_to_plot(ax, data, label):
    ax.plot(data[:, 0], data[:, 1], label=label)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 log_visualization.py <file_name> [<file_name> ...]")
        print("The program will plot the primal and dual residual of the given log files.")
        exit(0)
    file_names = sys.argv[1:]
    fig, axs = plt.subplots(2, 1)
    for file_name in file_names:
        data = get_data_from_file(file_name)
        k = data.shape[0] // 300
        axs[0].plot(data[::k, 0], data[::k, 1], label=file_name)
        axs[1].plot(data[::k, 0], data[::k, 2])
    axs[0].set_title("Primal Resudual")
    axs[1].set_title("Dual Residual")
    axs[0].set_ylim([0, 4000])
    axs[1].set_ylim([0, 2e-4])
    fig.legend()
    fig.set_size_inches(18.5, 10.5)
    plt.show()


