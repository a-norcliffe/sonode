#!/bin/bash

python plane_vibrations_anode.py --experiment_no 1
python plane_vibrations_anode_test.py --experiment_no 1
python plane_vibrations_anode.py --experiment_no 2
python plane_vibrations_anode_test.py --experiment_no 2
python plane_vibrations_anode.py --experiment_no 3
python plane_vibrations_anode_test.py --experiment_no 3

python plane_vibrations_sonode.py --experiment_no 1
python plane_vibrations_sonode_test.py --experiment_no 1
python plane_vibrations_sonode.py --experiment_no 2
python plane_vibrations_sonode_test.py --experiment_no 2
python plane_vibrations_sonode.py --experiment_no 3
python plane_vibrations_sonode_test.py --experiment_no 3

python make_errors.py --model 'anode' --experiment_no 1
python make_errors.py --model 'anode' --experiment_no 2
python make_errors.py --model 'anode' --experiment_no 3
python make_errors.py --model 'sonode' --experiment_no 1
python make_errors.py --model 'sonode' --experiment_no 2
python make_errors.py --model 'sonode' --experiment_no 3

python plot_figures.py












