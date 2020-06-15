import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc as rc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('--npoints', type=int, default=200)
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--model', type=str, default='sonode') #sonode or anode
parser.add_argument('--extra_dim', type=int, default=1)
args = parser.parse_args()




acc2_data = torch.load('data./test_position_data.pt')
acc2_data = acc2_data.detach().numpy().reshape(200)

if args.model == 'sonode':
    filename = args.model+'./'+str(args.experiment_no)+'./'
if args.model == 'anode':
    filename = args.model+'('+str(args.extra_dim)+')./'+str(args.experiment_no)+'./'

to_plot_acc2 = np.load(filename+'learnt_trajectory.npy')
samp_ts_array = np.arange(args.npoints)


###################################################
error = acc2_data - to_plot_acc2
window = 15
def moving_average(a, periods=window):
    weights = np.ones(periods) / periods
    return np.convolve(a, weights, mode='valid')

squared_error = error**2

error_ma = moving_average(squared_error)
error_ma = error_ma**0.5
samp_ts_array = samp_ts_array[:len(samp_ts_array)-window+1]
np.save(filename+'running_error.npy', error_ma)
np.save(filename+'running_error_times.npy', samp_ts_array)

rc('font', family='serif')
rc('text', usetex=True)
plt.plot(samp_ts_array, error_ma, label='SONODE', color='#004488')
plt.ylabel('Running Average RMSE', fontsize=16)
plt.xlabel('t', fontsize=16)
plt.title('F16 Running Error', fontsize=22)
plt.axvline(x=70, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=12, framealpha=1)

