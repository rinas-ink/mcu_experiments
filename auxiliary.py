import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree


def standardize_func(data, means, stds):
    return (data - means) / stds


def standardize(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1  # To avoid division by 0
    return standardize_func(data, means, stds), means, stds


def undo_standardize(data, means, stds):
    return data * stds + means



def get_eigen_decomposition(q):
    eigenvalues, eigenvectors = np.linalg.eig(q)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvectors, np.diag(sorted_eigenvalues)


def asymmetric_chamfer_dist(base_tree: KDTree, matching_points):
    """
    Distance from each point of matching cloud to the closest on base
    base_cloud: KD tree
    matching cloud: array of coordinates
    """
    dist, _ = base_tree.query(matching_points)
    return np.mean(dist)


def symmetric_chamfer_dist(points1, tree1: KDTree, points2, tree2: KDTree):
    return (asymmetric_chamfer_dist(tree1, points2) + asymmetric_chamfer_dist(tree2, points1)) / 2


def plot_heatmap(data):
    fig, ax = plt.subplots(figsize=(8, 6))

    cax = ax.imshow(data)

    fig.colorbar(cax)

    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))

    # ax.set_xticklabels(range(data.shape[1]))
    # ax.set_yticklabels(range(data.shape[0]))

    plt.show()
