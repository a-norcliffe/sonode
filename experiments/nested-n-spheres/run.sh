#!/bin/bash

python nested-n-spheres_anode.py --data_dimension 2 --extra_dim 0
python nested-n-spheres_anode.py --data_dimension 2 --extra_dim 1
python nested-n-spheres_sonode.py --data_dimension 2
python make_figure.py