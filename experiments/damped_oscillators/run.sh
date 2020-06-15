#!/bin/bash

python oscillators_node.py --experiment_no 1
python oscillators_node.py --experiment_no 2
python oscillators_node.py --experiment_no 3
python oscillators_anode.py --extra_dim 1 --experiment_no 1
python oscillators_anode.py --extra_dim 1 --experiment_no 2
python oscillators_anode.py --extra_dim 1 --experiment_no 3
python oscillators_sonode.py --experiment_no 1
python oscillators_sonode.py --experiment_no 2
python oscillators_sonode.py --experiment_no 3
python make_figure.py