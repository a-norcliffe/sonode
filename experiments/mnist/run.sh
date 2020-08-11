#!/bin/bash

python mnist_node.py --save './experiment_node1'
python mnist_node.py --save './experiment_node2'
python mnist_node.py --save './experiment_node3'
python mnist_sonode_conv_v.py --save './experiment_sonode_conv_v1'
python mnist_sonode_conv_v.py --save './experiment_sonode_conv_v2'
python mnist_sonode_conv_v.py --save './experiment_sonode_conv_v3'
python mnist_anode.py --extra_channels 1 --save'./experiment_anode1'
python mnist_anode.py --extra_channels 1 --save'./experiment_anode2'
python mnist_anode.py --extra_channels 1 --save'./experiment_anode3'
python make_errors.py
python plot_figures.py