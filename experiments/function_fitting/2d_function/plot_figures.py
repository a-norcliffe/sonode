import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import torch
import torch.nn as nn
import seaborn as sns
sns.set_style('dark')

gamma = np.array([0.1, 0.3])
omega = np.array([1, 1.2])

A, B, C, D = 1, 3, -5, 2
A_bar = np.array([omega[0]*B-gamma[0]*A, omega[1]*D-gamma[1]*C])
B_bar = np.array([gamma[0]*B+omega[0]*A, gamma[1]*D+omega[1]*C])

def x(t):
    return torch.exp(-gamma[0]*t)*(A*torch.cos(omega[0]*t)+B*torch.sin(omega[0]*t))
 
    
def real_vel_x(t):
    return torch.exp(-gamma[0]*t)*((A_bar[0])*torch.cos(omega[0]*t)-(B_bar[0])*torch.sin(omega[0]*t))


def y(t):
    return torch.exp(-gamma[1]*t)*(C*torch.cos(omega[1]*t)+D*torch.sin(omega[1]*t))


def real_vel_y(t):
    return torch.exp(-gamma[1]*t)*((A_bar[1])*torch.cos(omega[1]*t)-(B_bar[1])*torch.sin(omega[1]*t))

t0, tN = 0, 10
samp_ts = torch.linspace(t0, tN, 100).float()
samp_ts = torch.reshape(samp_ts, (100, 1))

x_ = x(samp_ts)
y_ = y(samp_ts)

x_ = x_.numpy()
x_ = np.transpose(x_)[0]

y_ = y_.numpy()
y_ = np.transpose(y_)[0]
    
x_vel_true = real_vel_x(samp_ts).numpy()
x_vel_true = np.transpose(x_vel_true)[0]

y_vel_true = real_vel_y(samp_ts).numpy()
y_vel_true = np.transpose(y_vel_true)[0]


rc('font', family='serif')
rc('text', usetex=True)
fig = plt.figure(figsize=[18, 4])
fig.subplots_adjust(hspace=0., wspace=0)



ax1 = plt.subplot(1, 4, 1)
#the final number in the filename is to choose the trajectory that looks best for the figure
filename = 'results./anode(2)./given_initial_conditions./1./'
x_pos = np.load(filename+'learnt_x.npy')
y_pos = np.load(filename+'learnt_y.npy')
vel_x = np.load(filename+'learnt_x_aug.npy')
vel_y = np.load(filename+'learnt_y_aug.npy')

plt.plot(x_pos, y_pos, label='Learnt func', color='#DDAA33', linewidth=2.8)
plt.plot(x_, y_, label='True func', color='#000000', linestyle='--', linewidth=2.8)
plt.plot(vel_x, vel_y, label='Learnt aug', color='#BB5566', linewidth=2.8)
plt.plot(x_vel_true, y_vel_true, label='True vel', color = '#004488', linestyle='--', linewidth=2.8)
#plt.legend(loc='upper center', ncol=2, borderaxespad=-4)
plt.title('ANODE(2)', fontsize=22, pad=26)
#plt.xlabel('x', fontsize=14)
#plt.ylabel('y', fontsize=14)
plt.xticks([])
plt.yticks([])




rc('font', family='serif')
rc('text', usetex=True)
ax2 = plt.subplot(1, 4, 2)
filename = 'results./anode(2)./given_initial_conditions./2./'
x_pos = np.load(filename+'learnt_x.npy')
y_pos = np.load(filename+'learnt_y.npy')
vel_x = np.load(filename+'learnt_x_aug.npy')
vel_y = np.load(filename+'learnt_y_aug.npy')

plt.plot(x_pos, y_pos, label='Learnt func', color='#DDAA33', linewidth=2.8)
plt.plot(x_, y_, label='True func', color='#000000', linestyle='--', linewidth=2.8)
plt.plot(vel_x, vel_y, label='Learnt aug', color='#BB5566', linewidth=2.8)
plt.plot(x_vel_true, y_vel_true, label='True vel', color = '#004488', linestyle='--', linewidth=2.8)
plt.legend(loc='upper center', ncol=2, borderaxespad=-4, fontsize=18)
#plt.title('ANODE(1)', fontsize=16)
#plt.xlabel('x', fontsize=14)
#plt.ylabel('y', fontsize=14)
plt.xticks([])
plt.yticks([])



rc('font', family='serif')
rc('text', usetex=True)
ax3 = plt.subplot(1, 4, 3)
filename = 'results./sonode./given_initial_conditions./1./'
x_pos = np.load(filename+'learnt_x.npy')
y_pos = np.load(filename+'learnt_y.npy')
vel_x = np.load(filename+'learnt_x_vel.npy')
vel_y = np.load(filename+'learnt_y_vel.npy')

plt.plot(x_pos, y_pos, label='Learnt func', color='#DDAA33', linewidth=2.8)
plt.plot(x_, y_, label='True func', color='#000000', linestyle='--', linewidth=2.8)
plt.plot(vel_x, vel_y, label='Learnt vel', color='#BB5566', linewidth=2.8)
plt.plot(x_vel_true, y_vel_true, label='True vel', color = '#004488', linestyle='--', linewidth=2.8)
#plt.legend(loc='upper center', ncol=2, borderaxespad=-4, fontsize=14)
plt.title('SONODE', fontsize=22, pad=26)
#plt.xlabel('x', fontsize=14)
#plt.ylabel('y', fontsize=14)
plt.xticks([])
plt.yticks([])



ax4 = plt.subplot(1, 4, 4)
filename = 'results./sonode./given_initial_conditions./2./'
x_pos = np.load(filename+'learnt_x.npy')
y_pos = np.load(filename+'learnt_y.npy')
vel_x = np.load(filename+'learnt_x_vel.npy')
vel_y = np.load(filename+'learnt_y_vel.npy')

plt.plot(x_pos, y_pos, label='Learnt func', color='#DDAA33', linewidth=2.8)
plt.plot(x_, y_, label='True func', color='#000000', linestyle='--', linewidth=2.8)
plt.plot(vel_x, vel_y, label='Learnt vel', color='#BB5566', linewidth=2.8)
plt.plot(x_vel_true, y_vel_true, label='True vel', color = '#004488', linestyle='--', linewidth=2.8)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.legend(loc='upper center', ncol=2, borderaxespad=-4, fontsize=18)
#plt.title('SONODE', fontsize=16)
#plt.xlabel('x', fontsize=14)
#plt.ylabel('y', fontsize=14)
plt.xticks([])
plt.yticks([])


plt.tight_layout()
plt.savefig('interpretability.png')
