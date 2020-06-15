import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import seaborn as sns

fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0)


##########################################
sns.set_style('dark')
rc('font', family='serif')
rc('text', usetex=True)
ax1 = plt.subplot(1,3,1)

noise = '0.3'
names = ['sonode', 'anode(1)']
labels =['SONODE', 'ANODE(1)']
colors = ['#004488', '#BB5566']


times = np.linspace(0, 15, 500)
def add_bit(x):
    loss_1 = np.load('results./'+noise+'./experiment1./'+names[x]+'./learnt_trajectory.npy')
    loss_2 = np.load('results./'+noise+'./experiment2./'+names[x]+'./learnt_trajectory.npy')
    loss_3 = np.load('results./'+noise+'./experiment3./'+names[x]+'./learnt_trajectory.npy')
    
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
    
    plt.plot(times, loss_mean, color=colors[x], label=labels[x])
    ax1.fill_between(x=times, y1=loss_p, y2=loss_m, alpha=0.2, color=colors[x])


plt.plot(times, np.sin(times), color='#DDAA33', label='sin(t)')
add_bit(0)
add_bit(1)

plt.legend(fontsize=12, loc='lower left')
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.axvline(x=10, linestyle='--', color='k')
plt.title('Noise = '+noise, fontsize=22)



#######################################################################
sns.set_style('dark')
ax2 = plt.subplot(1,3,2)

noise = '0.6'
names = ['sonode', 'anode(1)']
labels =['SONODE', 'ANODE(1)']
colors = ['#004488', '#BB5566']


times = np.linspace(0, 15, 500)
def add_bit(x):
    loss_1 = np.load('results./'+noise+'./experiment1./'+names[x]+'./learnt_trajectory.npy')
    loss_2 = np.load('results./'+noise+'./experiment2./'+names[x]+'./learnt_trajectory.npy')
    loss_3 = np.load('results./'+noise+'./experiment3./'+names[x]+'./learnt_trajectory.npy')
    
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
    
    plt.plot(times, loss_mean, color=colors[x], label=labels[x])
    ax2.fill_between(x=times, y1=loss_p, y2=loss_m, alpha=0.2, color=colors[x])


plt.plot(times, np.sin(times), color='#DDAA33', label='sin(t)')
add_bit(0)
add_bit(1)
rc('font', family='serif')
rc('text', usetex=True)

plt.legend(fontsize=12, loc='lower left')
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.axvline(x=10, linestyle='--', color='k')
plt.title('Noise = '+noise, fontsize=22)













################################
sns.set_style('darkgrid')
ax3 = plt.subplot(1, 3, 3)
    
colours = ['#004488','#BB5566']
markers = ['D', 'o']
labels = ['SONODE', 'ANODE(1)']


sampled_results = np.load('results./results.npy')

plotting_results = np.empty((2, 2, 2, 8))

for i in range(2):
    for j in range(2):
        for k in range(8):
            plotting_results[i][j][0][k] = np.mean(sampled_results[i][j][k])
            plotting_results[i][j][1][k] = np.std(sampled_results[i][j][k])
            

noise = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

#def plot_train(x):
#    data = plotting_results[0][x]
#    mean = data[0]
#    error = data[1]
#    plt.scatter(noise, mean, color=colours[x], marker=markers[x],\
#                s=40, label=labels[x])
#    plt.errorbar(noise, mean, yerr=error, color=colours[x], linewidth=1,\
#                 linestyle='None', capsize=4, capthick = 1)
    

def plot_test(x):
    data = plotting_results[1][x]
    mean = data[0]
    error = data[1]
    plt.scatter(noise, mean, color=colours[x], marker=markers[x],\
                s=40, label=labels[x])#, alpha=0.7)
    plt.errorbar(noise, mean, yerr=error, color=colours[x], linewidth=1,\
                 linestyle='None', capsize=4, capthick = 1)#, alpha=0.7)
    
    
    
plot_test(0)
plot_test(1)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.xlabel('Noise', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.title('Test MSE', fontsize=20)
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],\
           loc='upper left', fontsize=14)

plt.tight_layout()
plt.savefig('test_loss_with_noise.png', bbox_inches='tight')




