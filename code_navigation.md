# What is where

1. [simple_angle_pipeline.ipynb](simple_angle_pipeline.ipynb), [cylinder_pipeline.ipynb](cylinder_pipeline.ipynb), [roll_pipeline.ipynb](roll_pipeline.ipynb) - pipelines with corresponding figures.
2. [simple_angles_dataset_generator.py](simple_angles_dataset_generator.py), [cylinder_dataset_generator.py](cylinder_dataset_generator.py), [swiss_roll_dataset_generator.py](swiss_roll_dataset_generator.py) - corresponding generators of figres. You may run mains of generators to see how they look like. [dataset_generator.py](dataset_generator.py)  -common functions for these generators.
3. [mcu.py](mcu.py) - the algorithm from the article and functions that test and plot different parts of it.
4. Now angles look like folded piece of paper, where we vary angle in between and rotation of a whole figure. Old angles that looked like and edge of cube weren't refactored and probably won't run. Files, that aren't listed above are releted to old angles.