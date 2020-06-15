import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import matplotlib.axes
import seaborn as sns

#########################################################################
sns.set_style('darkgrid')
rc('font', family='serif')
rc('text', usetex=True) 
iters = np.load('sonode./1./itr_arr.npy')
sonode_1 = np.load('sonode./1./loss_arr.npy')
sonode_2 = np.load('sonode./2./loss_arr.npy')
sonode_3 = np.load('sonode./3./loss_arr.npy')

sonode_loss = np.empty((len(sonode_1), 3))
for i in range(len(sonode_1)):
    sonode_loss[i][0] = sonode_1[i]
    sonode_loss[i][1] = sonode_2[i]
    sonode_loss[i][2] = sonode_3[i]
    
sonode_mean = np.empty(len(sonode_1))
for i in range(len(sonode_1)):
    sonode_mean[i] = np.mean(sonode_loss[i])

sonode_std = np.empty(len(sonode_1))
for i in range(len(sonode_1)):
    sonode_std[i] = np.std(sonode_loss[i])

sonode_p = sonode_mean+sonode_std
sonode_m = sonode_mean-sonode_std


fig, ax = plt.subplots()
plt.plot(iters, sonode_mean, color='#004488', label='SONODE')
ax.fill_between(x=iters, y1=sonode_p, y2=sonode_m, alpha=0.2, color='#004488')

####################################################################

anode_1 = np.load('anode(1)./1./loss_arr.npy')
anode_2 = np.load('anode(1)./2./loss_arr.npy')
anode_3 = np.load('anode(1)./3./loss_arr.npy')

anode_loss = np.empty((len(anode_1), 3))
for i in range(len(anode_1)):
    anode_loss[i][0] = anode_1[i]
    anode_loss[i][1] = anode_2[i]
    anode_loss[i][2] = anode_3[i]
    
anode_mean = np.empty(len(anode_1))
for i in range(len(anode_1)):
    anode_mean[i] = np.mean(anode_loss[i])

anode_std = np.empty(len(anode_1))
for i in range(len(anode_1)):
    anode_std[i] = np.std(anode_loss[i])

anode_p = anode_mean+anode_std
anode_m = anode_mean-anode_std

plt.plot(iters, anode_mean, color='#BB5566', label='ANODE(1)')
ax.fill_between(x=iters, y1=anode_p, y2=anode_m, alpha=0.2, color='#BB5566')
                
                
                
                
####################################################################

node_1 = np.load('node./1./loss_arr.npy')
node_2 = np.load('node./2./loss_arr.npy')
node_3 = np.load('node./3./loss_arr.npy')

node_loss = np.empty((len(node_1), 3))
for i in range(len(node_1)):
    node_loss[i][0] = node_1[i]
    node_loss[i][1] = node_2[i]
    node_loss[i][2] = node_3[i]
    
node_mean = np.empty(len(node_1))
for i in range(len(node_1)):
    node_mean[i] = np.mean(node_loss[i])

node_std = np.empty(len(node_1))
for i in range(len(node_1)):
    node_std[i] = np.std(node_loss[i])

node_p = node_mean+node_std
node_m = node_mean-node_std

plt.plot(iters, node_mean, color='#DDAA33', label='NODE')
ax.fill_between(x=iters, y1=node_p, y2=node_m, alpha=0.2, color='#DDAA33')                
                
rc('font', family='serif')
rc('text', usetex=True)                                
plt.xlim(0, 200)
plt.ylim(0, 1.3)
plt.legend(fontsize=12)
plt.title('Damped Oscillators Training MSE', fontsize=20)
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.savefig('damped_oscillators_loss.png', bbox_inches='tight')




















