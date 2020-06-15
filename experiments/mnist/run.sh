#!/bin/bash

python mnist_node.py --save './experiment_node1'
python mnist_node.py --save './experiment_node2'
python mnist_node.py --save './experiment_node3'
python mnist_sonode_conv_v.py --save './experiment_sonode_conv_v1'
python mnist_sonode_conv_v.py --save './experiment_sonode_conv_v2'
python mnist_sonode_conv_v.py --save './experiment_sonode_conv_v3'
python making_errors.py
python plot_figures.py