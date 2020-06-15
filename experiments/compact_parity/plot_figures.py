import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import seaborn as sns
sns.set_style('dark')
rc('font', family='serif')
rc('text', usetex=True)

node_ts = np.load('node./ts.npy')
node_t1 = np.load('node./trajectory_1.npy')
node_t2 = np.load('node./trajectory_2.npy')

anode_ts = np.load('anode(1)./ts.npy')
anode_t1 = np.load('anode(1)./trajectory_1.npy')
anode_t2 = np.load('anode(1)./trajectory_2.npy')

sonode_ts = np.load('sonode./ts.npy')
sonode_t1 = np.load('sonode./trajectory_1.npy')
sonode_t2 = np.load('sonode./trajectory_2.npy')

fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0)

ax1 = plt.subplot(1, 3, 1)
plt.plot(node_ts, node_t1, color='#004488', linewidth=2.2)
plt.plot(node_ts, node_t2, color='#BB5566', linewidth=2.2)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.title('NODE', fontsize=20)

ax2 = plt.subplot(1, 3, 2)
plt.plot(anode_ts, anode_t1, color='#004488', linewidth=2.2)
plt.plot(anode_ts, anode_t2, color='#BB5566', linewidth=2.2)
rc('font', family='serif')
rc('text', usetex=True)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.title('ANODE(1)', fontsize=20)

ax3 = plt.subplot(1, 3, 3)
plt.plot(sonode_ts, sonode_t1, color='#004488', linewidth=2.2)
plt.plot(sonode_ts, sonode_t2, color='#BB5566', linewidth=2.2)
rc('font', family='serif')
rc('text', usetex=True)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.title('SONODE', fontsize=20)

plt.tight_layout()
#sns.set_style('white')
plt.savefig('compact_parity.png', bbox_inches='tight')