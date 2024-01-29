import unittest
import numpy as np

import dataset_generator
from dataset_generator import get_control_vars
import swiss_roll_dataset_generator as roll_gen
from mcu import construct_graph, k_nearest_neighbours

class GraphStructureTestCase(unittest.TestCase):
    def test_graph_structure(self):
        np.random.seed(179)
        for test_n in range(10):
            k = np.random.randint(2, 10)
            n = np.random.randint(k + 1, 100)
            control_vars = get_control_vars(deterministic=False, dimensionality=roll_gen.get_p(),
                                                               size=n, lw=[1, 1], up=[10, 10])
            ys = dataset_generator.generate_array_of_figures(control_vars, roll_gen.generate_swiss_roll, noise_level=0.1, min_num_points=1600)
            edges = construct_graph(ys, k)

            degree = [0] * n
            for edge in edges:
                degree[edge[0]] += 1
                degree[edge[1]] += 1
            
            for i in range(n):
                self.assertGreaterEqual(degree[i], k, "Node {} has degree {} in test {}".format(i, degree[i], test_n))
                self.assertLessEqual(degree[i], n - 1, "Node {} has degree {} in test {}".format(i, degree[i], test_n))

            edges = set(map(tuple, edges))
            for y in ys:
                neighbours = sorted(k_nearest_neighbours(ys, y, k))
                for i in range(k + 1):
                    for j in range(i + 1, k + 1):
                        self.assertIn((neighbours[i], neighbours[j]), edges, "Edge ({}, {}) not found in test {}".format(neighbours[i], neighbours[j], test_n))



    def test_on_sample(self):
        ys = np.array([[1, 1], [-5, 3.33], [0, -2], [2, 1.5], [6, 4.1], [-1, 7], [-1, 0], [3, 5.05], [1, 6], [1, 3]])
        k = 2
        edges = construct_graph(ys, k)
        # plot_two_embeddings_with_edges(ys, ys, edges)

        correct_edges = [[0, 2], [0, 3], [0, 6], [0, 9], [1, 5], [1, 6], [2, 6], [3, 4], [3, 7], [3, 9], [4, 7], [5, 6], [5, 7], [5, 8], [7, 8], [7, 9], [8, 9]]
        self.assertListEqual(sorted(edges.tolist()), sorted(correct_edges))



if __name__ == '__main__':
    unittest.main()