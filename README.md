
# Experiment setup for Maximum Covariance Unfolding Regression

In the following repository there is implementation of MCU method from the corresponding article 
[Maximum Covariance Unfolding Regression: A Novel Covariate-based Manifold Learning Approach for Point Cloud Data](https://arxiv.org/abs/2303.17852) 
by Qian Wang and Kamran Paynabar.

The original method is implemented in `MCUOriginalModel` class 
([`mcu_original.py`](mcu_original.py)).

The modified version with distance metric replaced by Chamfer distance is implemented in `MCUChamferModel` class
([`mcu_chamfer.py`](mcu_chamfer.py)).

Dataset generators for cylinders, angles and Swiss rolls are in [cylinder_dataset_generator.py](cylinder_dataset_generator.py),
[simple_angles_dataset_generator.py](simple_angles_dataset_generator.py) and [swiss_roll_dataset_generator.py](swiss_roll_dataset_generator.py)
correspondingly.

Whole pipelines of original and modified methods can be found in python notebooks `<method>_<figure_type>_pipeline_<parameter dimentionality>d.ipynb`.