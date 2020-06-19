import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import seaborn as sns

sns.set_style('dark')
rc('font', family='serif')
rc('text', usetex=True)

names = ['node./', 'anode(1)./', 'sonode./']
def plot_exp(x):
    filename = names[x]
    ts = np.load(filename+'ts.npy')
    z_real = np.load(filename+'z_real.npy')
    samp_ts = np.load(filename+'samp_ts.npy')
    measured_z = np.load(filename+'measured_z.npy')
    test_ts = np.load(filename+'test_ts.npy')
    test_z = np.load(filename+'test_z.npy')
    learnt_ts = np.load(filename+'learnt_ts.npy')
    learnt_trajectory = np.load(filename+'learnt_trajectory.npy')
    
    plt.scatter(samp_ts, measured_z, label='Sampled', color='k', s=30)
    plt.scatter(test_ts, test_z, marker='x', color='k', label='Test', s=70)
    plt.plot(learnt_ts, learnt_trajectory, color='#BB5566', label='Learnt', linewidth=2)
    plt.plot(ts, z_real, color='#004488', label='exp(0.1667t)', linestyle='--', linewidth=2)
    
fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0)
   
ax1 = plt.subplot(1,3,1)
plot_exp(0)
plt.xlabel('t', fontsize=18)
plt.ylabel('x', fontsize=18)
plt.legend(loc='upper left', fontsize = 14)
plt.title('NODE', fontsize=22)

ax2 = plt.subplot(1, 3, 2)
plot_exp(1)
rc('font', family='serif')
rc('text', usetex=True)
plt.xlabel('t', fontsize=18)
plt.ylabel('x', fontsize=18)
plt.legend(loc='upper left', fontsize = 14)
plt.title('ANODE(1)', fontsize=22)

ax3 = plt.subplot(1, 3, 3)
plot_exp(2)
rc('font', family='serif')
rc('text', usetex=True)
plt.xlabel('t', fontsize=18)
plt.ylabel('x', fontsize=18)
plt.legend(loc='upper left', fontsize = 14)
plt.title('SONODE', fontsize=22)

plt.tight_layout()
plt.savefig('interpolation.png', bbox_inches='tight')






