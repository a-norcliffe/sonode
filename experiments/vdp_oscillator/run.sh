#!/bin/bash

python vdp_anode.py --experiment_no 1
python vdp_anode_test.py --experiment_no 1
python vdp_anode.py --experiment_no 2
python vdp_anode_test.py --experiment_no 2
python vdp_anode.py --experiment_no 3
python vdp_anode_test.py --experiment_no 3

python vdp_sonode.py --experiment_no 1
python vdp_sonode_test.py --experiment_no 1
python vdp_sonode.py --experiment_no 2
python vdp_sonode_test.py --experiment_no 2
python vdp_sonode.py --experiment_no 3
python vdp_sonode_test.py --experiment_no 3

python make_errors.py --model 'anode' --experiment_no 1
python make_errors.py --model 'anode' --experiment_no 2
python make_errors.py --model 'anode' --experiment_no 3
python make_errors.py --model 'sonode' --experiment_no 1
python make_errors.py --model 'sonode' --experiment_no 2
python make_errors.py --model 'sonode' --experiment_no 3

python plot_figures.py








