import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import seaborn as sns




fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0)



######################################

sns.set_style('darkgrid')
rc('font', family='serif')
rc('text', usetex=True)
ax1 = plt.subplot(1,3,1)


names = ['experiment_sonode_conv_v', 'experiment_node']
labels =['SONODE', 'NODE']
colors = ['#004488', '#BB5566']
to_plot_names = ['epoch_arr.npy', 'running_train_acc.npy', 'running_test_acc.npy', 'nfe_b_arr.npy', 'nfe_f_arr.npy', 'time_avg_arr.npy', 'time_val_arr.npy']

def add_bit(x, to_plot):
    iters = np.load(names[x]+'1./running_epoch_arr.npy')
    loss_1 = np.load(names[x]+'1./'+to_plot_names[to_plot])
    loss_2 = np.load(names[x]+'2./'+to_plot_names[to_plot])
    loss_3 = np.load(names[x]+'3./'+to_plot_names[to_plot])
    
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




add_bit(0, 1)    
add_bit(1, 1)
rc('font', family='serif')
rc('text', usetex=True)
plt.legend(loc='lower right', fontsize=12)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Running Training Accuracy', fontsize=16)
plt.ylim(0.994, 1.0002)
plt.title('MNIST Training Accuracy', fontsize=22)


####################################################

ax2 = plt.subplot(1, 3, 2)


def add_bit(x, to_plot):
    iters = np.load(names[x]+'1./running_epoch_arr.npy')
    loss_1 = np.load(names[x]+'1./'+to_plot_names[to_plot])
    loss_2 = np.load(names[x]+'2./'+to_plot_names[to_plot])
    loss_3 = np.load(names[x]+'3./'+to_plot_names[to_plot])
    
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
    ax2.fill_between(x=iters, y1=loss_p, y2=loss_m, alpha=0.2, color=colors[x])




add_bit(0, 2)
add_bit(1, 2)
rc('font', family='serif')
rc('text', usetex=True)
#plt.legend(loc='lower right', fontsize=12)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Running Test Accuracy', fontsize=16)
plt.ylim(0.991, 0.997)
plt.title('MNIST Test Accuracy', fontsize=22)



##########################################################

ax3 = plt.subplot(1, 3, 3)


def add_bit(x, to_plot):
    iters = np.load(names[x]+'1./epoch_arr.npy')
    loss_1 = np.load(names[x]+'1./'+to_plot_names[to_plot])
    loss_2 = np.load(names[x]+'2./'+to_plot_names[to_plot])
    loss_3 = np.load(names[x]+'3./'+to_plot_names[to_plot])
    
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




add_bit(0, 4)
add_bit(1, 4)
rc('font', family='serif')
rc('text', usetex=True)
#plt.legend(fontsize=12)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('NFE', fontsize=16)
plt.ylim(19, 27)
plt.title('MNIST NFE', fontsize=22)


plt.tight_layout()
plt.savefig('mnist.png', bbox_inches='tight')