#!/bin/bash

python 2d_function_fit_anode_given_init_conditions.py --experiment_no 1
python 2d_function_fit_anode_given_init_conditions.py --experiment_no 2
python 2d_function_fit_sonode_given_init_conditions.py --experiment_no 1
python 2d_function_fit_sonode_given_init_conditions.py --experiment_no 2
python plot_figures.py
python 2d_function_fit_anode_start_at_zero.py --extra_dim 1 --experiment_no 1
python plot_figure_anode1.py