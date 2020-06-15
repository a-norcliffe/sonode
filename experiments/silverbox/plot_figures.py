import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('--npoints', type=int, default=5000)
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
v2_tensor =torch.tensor(v2_data).float()
v2_tensor = v2_tensor.reshape(args.npoints, 1)

samp_ts_array = np.arange(args.npoints)
samp_ts = torch.tensor(samp_ts_array).float()
samp_ts = samp_ts.reshape(args.npoints, 1)



fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0)



######################################

sns.set_style('darkgrid')
rc('font', family='serif')
rc('text', usetex=True)
ax1 = plt.subplot(1,3,1)


names = ['sonode', 'anode(1)']
labels =['SONODE', 'ANODE(1)']
colors = ['#004488', '#BB5566']

def add_bit(x):
    iters = np.load(names[x]+'./1./itr_arr.npy')
    loss_1 = np.load(names[x]+'./1./loss_arr.npy')
    loss_2 = np.load(names[x]+'./2./loss_arr.npy')
    loss_3 = np.load(names[x]+'./3./loss_arr.npy')
    
    loss = np.empty((len(loss_1),3))
    for i in range(len(loss_1)):
        loss[i][0] = loss_1[i]
        loss[i][1] = loss_2[i]
        loss[i][2] = loss_3[i]
    
    loss_mean = np.empty(len(loss_1))
    for i in range(len(loss_1)):
        loss_mean[i] = np.mean(loss[i])
    
    loss_std = np.empty(len(loss_1))
    for i in range(len(loss_1)):
        loss_std[i] = np.std(loss[i])
        
    loss_p = loss_mean + loss_std
    loss_m = loss_mean - loss_std
    
    plt.plot(iters, loss_mean, color=colors[x], label=labels[x])
    ax1.fill_between(x=iters, y1=loss_p, y2=loss_m, alpha=0.2, color=colors[x])



add_bit(0)
add_bit(1)
rc('font', family='serif')
rc('text', usetex=True)
plt.legend(fontsize=12)
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.ylim(0, 1.2)
plt.title('Silverbox Training MSE', fontsize=22)








############################
sns.set_style('dark')
ax2 = plt.subplot(1, 3, 2)
sonode_to_plot_v2 = np.load('sonode./3./v2_test.npy')
anode_to_plot_v2 = np.load('anode(1)./3./v2_test.npy')
plt.plot(samp_ts_array, v2_data, label='True $V_{2}$', color='#DDAA33')
plt.plot(samp_ts_array, sonode_to_plot_v2, label='SONODE', color='#004488')
plt.plot(samp_ts_array, anode_to_plot_v2, label='ANODE(1)', color='#BB5566')
rc('font', family='serif')
rc('text', usetex=True)
plt.xlabel('t', fontsize=16)
plt.ylabel('$V_{2}$', fontsize=16)
plt.legend(loc='upper left', fontsize=12, framealpha=1)
plt.axvline(x=1000, linestyle='--', color='k')
plt.title('Silverbox $V_{2}$', fontsize=22)

 

##########################
sns.set_style('darkgrid')
ax3 = plt.subplot(1, 3, 3)


names = ['sonode', 'anode(1)']
labels =['SONODE', 'ANODE(1)']
colors = ['#004488', '#BB5566']

def add_bit(x):
    iters = np.load(names[x]+'./1./running_error_times.npy')
    loss_1 = np.load(names[x]+'./1./running_error.npy')
    loss_2 = np.load(names[x]+'./2./running_error.npy')
    loss_3 = np.load(names[x]+'./3./running_error.npy')
    
    loss = np.empty((len(loss_1),3))
    for i in range(len(loss_1)):
        loss[i][0] = loss_1[i]
        loss[i][1] = loss_2[i]
        loss[i][2] = loss_3[i]
    
    loss_mean = np.empty(len(loss_1))
    for i in range(len(loss_1)):
        loss_mean[i] = np.mean(loss[i])
    
    loss_std = np.empty(len(loss_1))
    for i in range(len(loss_1)):
        loss_std[i] = np.std(loss[i])
        
    loss_p = loss_mean + loss_std
    loss_m = loss_mean - loss_std
    
    plt.plot(iters, loss_mean, color=colors[x], label=labels[x])
    ax3.fill_between(x=iters, y1=loss_p, y2=loss_m, alpha=0.2, color=colors[x])


add_bit(0)
add_bit(1)
rc('font', family='serif')
rc('text', usetex=True)
plt.ylabel('Running Average RMSE', fontsize=16)
plt.xlabel('t', fontsize=16)
plt.title('Silverbox Running Error', fontsize=22)
plt.axvline(x=1000, linestyle='--', color='k')
plt.legend(loc='upper left', fontsize=12, framealpha=1)


plt.tight_layout()
plt.savefig('silverbox.png', bbox_inches='tight')