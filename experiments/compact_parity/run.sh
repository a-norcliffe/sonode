#!/bin/bash

python compact_parity_anode.py --extra_dim 0
python compact_parity_anode.py --extra_dim 1
python compact_parity_sonode.py
python plot_figures.py