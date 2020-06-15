import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc as rc
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--npoints', type=int, default=5000)
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--model', type=str, default='anode') #sonode or anode
parser.add_argument('--extra_dim', type=int, default=1)
args = parser.parse_args()




data = pd.read_csv('SNLS80mv.csv', header=0, nrows=args.npoints)

data = data.values.tolist()
data = np.asarray(data)
data = np.transpose(data)

v1_data = data[0]
v2_data = data[1]

v1_data = v1_data - np.full_like(v1_data, np.mean(v1_data))
v2_data = v2_data - np.full_like(v2_data, np.mean(v2_data))
rescaling = 100
v1_data = rescaling*v1_data
v2_data = rescaling*v2_data

samp_ts_array = np.arange(args.npoints)
samp_ts_array = samp_ts_array.reshape(args.npoints, 1)



if args.model == 'sonode':
    filename = args.model+'./'+str(args.experiment_no)+'./'
if args.model == 'anode':
    filename = args.model+'('+str(args.extra_dim)+')./'+str(args.experiment_no)+'./'

to_plot_acc2 = np.load(filename+'v2_test.npy')
samp_ts_array = np.arange(args.npoints)


###################################################
error = v2_data - to_plot_acc2
window = 300
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
plt.plot(samp_ts_array, error_ma, label=args.model, color='#004488')
plt.ylabel('Running Average RMSE', fontsize=16)
plt.xlabel('t', fontsize=16)
plt.title('Silverbox Running Error', fontsize=22)
plt.axvline(x=1000, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=12, framealpha=1)