import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from matplotlib.pyplot import rc as rc

sonode_learnt_trajectory = np.load('sonode./1./learnt_trajectory.npy')
anode_learnt_trajectory = np.load('anode(1)./1./learnt_trajectory.npy')
test_position_data = torch.load('data./test_position_data.pt')
test_position_data = test_position_data.detach().numpy().reshape(len(sonode_learnt_trajectory))
test_time_data = torch.load('data./test_time_data.pt')
test_time_data = test_time_data.detach().numpy().reshape(len(sonode_learnt_trajectory))



fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0)


######################################
sns.set_style('darkgrid')
rc('font', family='serif')
rc('text', usetex=True)
ax1 = plt.subplot(1, 3, 1)

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
plt.ylim(0, 7)

plt.title('VDP Oscillator Training MSE', fontsize=22)



########################################
sns.set_style('dark')
ax2 = plt.subplot(1, 3, 2)
plt.plot(test_time_data, test_position_data, label='True VDP', color='#DDAA33')
plt.plot(test_time_data, sonode_learnt_trajectory, label='SONODE', color='#004488')
plt.plot(test_time_data, anode_learnt_trajectory, label='ANODE(1)', color='#BB5566')
rc('font', family='serif')
rc('text', usetex=True)
plt.axvline(x=70, linestyle='--', color='k')
plt.xlabel('t', fontsize=16)
plt.ylabel('$x$', fontsize=16)
plt.legend(loc='lower right', fontsize=12, framealpha=1)
plt.title('VDP Oscillator Displacement', fontsize=22)





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
plt.title('VDP Oscillator Running Error', fontsize=22)
plt.axvline(x=70, linestyle='--', color='k')
plt.legend(loc='upper left', fontsize=12, framealpha=1)



plt.tight_layout()
plt.savefig('vdp.png', bbox_inches='tight')





