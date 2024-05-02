import numpy as np

import auxiliary
import dataset_generator
from mcu_base import MCUbase
from mcu_original import MCUOriginalModel


class MCUexperiments:

    def __init__(self, mcu_model: MCUOriginalModel):
        self.mcu_model: MCUbase = mcu_model

    def test_predictive_optimization(self, lw, up, figure_generator, figure_point_cnt, noise_level=0, pieces_cnt=10,
                                     test_data_size=50, same_value=False, gd = False):
        p = self.mcu_model.params.shape[1]
        intervals = [np.linspace(lw[i], up[i], pieces_cnt + 1) for i in range(p)]
        interval_runs_shape = tuple([pieces_cnt] * p + [p + 1, 2])
        interval_runs = np.empty(shape=interval_runs_shape)

        for indices in np.ndindex(*[pieces_cnt] * p):
            interval_lw = np.array([intervals[dim][index] for dim, index in enumerate(indices)])
            interval_up = np.array([intervals[dim][index + 1] for dim, index in enumerate(indices)])

            if same_value:
                interval_lw = interval_up

            test_control_vars = dataset_generator.get_control_vars(deterministic=False,
                                                                   dimensionality=p,
                                                                   size=test_data_size,
                                                                   lw=interval_lw, up=interval_up)
            test_control_vars_dict = dataset_generator.put_control_vars_in_dict(test_control_vars, p,
                                                                                self.mcu_model.params_names)

            test_figures = dataset_generator.generate_array_of_figures(test_control_vars_dict, figure_generator,
                                                                       noise_level=noise_level,
                                                                       min_num_points=figure_point_cnt)
            x_opts = []
            for (figure, control_var) in zip(test_figures, test_control_vars):
                x_opt, x_err = self.mcu_model.predict(figure, gd = gd)
                x_opts.append(x_opt)
                print("-----------")
                print(f"x_opt  = {x_opt}, x_err = {x_err}")
                print(f"x_real = {control_var}")

            x_opts = np.array(x_opts)
            test_control_vars = np.array(test_control_vars)
            errors = x_opts - test_control_vars
            errors_common = np.linalg.norm(errors, axis=1)

            interval_runs[indices] = [
                                         [np.median(abs(errors[:, dim])),
                                          np.percentile(abs(errors[:, dim]), 75) - np.percentile(abs(errors[:, dim]),
                                                                                                 25)]
                                         for dim in range(p)
                                     ] + [[np.median(errors_common),
                                           np.percentile(errors_common, 75) - np.percentile(errors_common, 25)]]

            for dim in range(p):
                print(f'errors{dim} = {abs(errors[:, dim])}')

        return interval_runs, np.array(intervals)
