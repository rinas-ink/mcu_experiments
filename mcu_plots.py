import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import dataset_generator
from mcu_original import MCUOriginalModel


class MCUplots:
    def __init__(self, mcu_model: MCUOriginalModel):
        self.mcu_model = mcu_model

    def plot_embeddings_vs_parameters(self, with_edges=False, annotate=False):
        params = self.mcu_model.original_params
        params_names = self.mcu_model.params_names
        embedding = self.mcu_model.embedded_y_as_params()
        edges = self.mcu_model.graph_edges
        _, params_cnt = np.shape(params)
        fig, axs = plt.subplots(params_cnt * (params_cnt - 1) // 2, 2,
                                figsize=(14, params_cnt * (params_cnt - 1) * 7 / 2))
        if axs.ndim == 1:
            axs = np.array([axs])
        cnt = 0
        for i in range(1, params_cnt):
            for j in range(i):

                axs[cnt, 0].scatter(params[:, j], params[:, i], s=10, c=params[:, j], cmap=plt.cm.Spectral)
                axs[cnt, 0].set_title('Parameters')
                axs[cnt, 0].set_xlabel(params_names[j])
                axs[cnt, 0].set_ylabel(params_names[i])

                axs[cnt, 1].scatter(embedding[:, j], embedding[:, i], s=10, c=embedding[:, j], cmap=plt.cm.Spectral)
                axs[cnt, 1].set_title('Parameters, computed from embedding')
                axs[cnt, 1].set_xlabel(params_names[j])
                axs[cnt, 1].set_ylabel(params_names[i])
                if annotate:
                    for index, (x, y) in enumerate(zip(params[:, j], params[:, i])):
                        if index > 100:
                            break
                        axs[cnt, 0].annotate(str(index), (x, y), textcoords="offset points", xytext=(0, 10),
                                             ha='center')
                    for index, (x, y) in enumerate(zip(embedding[:, j], embedding[:, i])):
                        if index > 100:
                            break
                        axs[cnt, 1].annotate(str(index), (x, y), textcoords="offset points", xytext=(0, 10),
                                             ha='center')

                if with_edges == False:
                    edge_colors = ['red', 'green', 'blue']
                    edge_colors = [edge_colors[np.random.randint(0, 3)] for _ in range(len(edges))]
                    params_seg = np.hstack((params[edges[:, 0]][:, [j, i]], params[edges[:, 1]][:, [j, i]]))
                    params_seg = params_seg.reshape((-1, 2, 2))
                    rec_edges = LineCollection(params_seg, colors=edge_colors, alpha=0.5)
                    axs[cnt, 0].add_collection(rec_edges)

                    ld_segments = np.hstack((embedding[edges[:, 0]][:, [j, i]], embedding[edges[:, 1]][:, [j, i]]))
                    ld_segments = ld_segments.reshape((-1, 2, 2))
                    ld_edges = LineCollection(ld_segments, colors=edge_colors, alpha=0.5)
                    axs[cnt, 1].add_collection(ld_edges)

                cnt += 1
        plt.show()

    def plot_2d_predictive_optimization_heatmaps(self, intervals, interval_runs,
                                                 fixed_params_map=None, filename=None, title=""):
        """
        :param intervals: interval split points for each parameter
        :param interval_runs: interval runs from `test_predictive_optimization`
        :param fixed_params_map: dictionary, that represents which parameter at which piece we fix.
        For example {0:1}, when p=3, means that we make this slice of `interval_runs`: `interval_runs[1, :, :]`.
        """
        if fixed_params_map is None:
            fixed_params_map = dict()
        fixed_indices = [slice(None)] * interval_runs.ndim
        for dim_index, value in fixed_params_map.items():
            fixed_indices[dim_index] = value
        interval_runs = interval_runs[tuple(fixed_indices)]
        pieces_cnt = interval_runs.shape[0]
        params_names = self.mcu_model.params_names
        p = self.mcu_model.params.shape[1]

        remaining_params_idx = np.setdiff1d(np.arange(p), np.array(list(fixed_params_map.keys()), dtype=int))
        if len(remaining_params_idx) != 2:
            raise "There should be exactly 2 non-fixed parameters left"
        axes_param_names = params_names[remaining_params_idx]

        fig, axs = plt.subplots(p + 1, 2, figsize=(16, 8 * (p + 1)))
        if fixed_params_map != {}:
            plot_title = f"{title}, Fixed parameters: "
        else:
            plot_title = f"{title}"
        for param, bound in fixed_params_map.items():
            plot_title = plot_title + params_names[
                param] + f' in [{intervals[param][bound], intervals[param][bound + 1]}]; '

        fig.suptitle(plot_title, fontsize=16)

        imgs = []

        for l in range(2):
            for k in range(p + 1):
                imgs.append(axs[k, l].imshow(interval_runs[:, :, k, l], cmap='YlGnBu', interpolation='nearest'))
                axs[k, l].set_xlabel(axes_param_names[0])
                axs[k, l].set_ylabel(axes_param_names[1])
                axs[k, l].set_xticks(np.arange(pieces_cnt + 1) - 0.5, [f'{x:.1f}' for x in intervals[0]])
                axs[k, l].set_yticks(np.arange(pieces_cnt + 1) - 0.5, [f'{y:.1f}' for y in intervals[1]])
                fig.colorbar(imgs[-1], ax=axs[k, l])

                for i in range(pieces_cnt):
                    for j in range(pieces_cnt):
                        axs[k, l].text(j, i, f'{interval_runs[i, j, k, l]:.1f}', ha='center', va='center',
                                       color='black')

        for k in range(p):
            axs[k, 0].set_title(f'{params_names[k]} Error')
            axs[k, 1].set_title(f'{params_names[k]} IQR')
        axs[p, 0].set_title('Norm of common error')
        axs[p, 1].set_title('Norm of common IQR')

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename)

        plt.show()
