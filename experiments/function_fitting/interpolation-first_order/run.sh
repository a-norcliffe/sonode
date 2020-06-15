#!/bin/bash

python interpolation_anode.py --extra_dim 0
python interpolation_anode.py --extra_dim 1
python interpolation_sonode.py
python plot_figures.py