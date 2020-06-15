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

filename = 'results./anode(1)./start_at_zero./1./'
x_pos = np.load(filename+'learnt_x.npy')
y_pos = np.load(filename+'learnt_y.npy')
aug = np.load(filename+'learnt_aug.npy')
samp_ts = np.linspace(t0, tN, 100)


fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0)


sns.set_style('dark')
rc('font', family='serif')
rc('text', usetex=True)
ax1 = plt.subplot(1, 3, 1)


plt.plot(x_pos, y_pos, color='#BB5566', label='Learnt func', linewidth=2.7)
plt.plot(x_, y_, color = '#004488', linestyle='--', label='True func', linewidth=2)
plt.legend(loc='lower left', fontsize=14)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.title('ANODE(1) 2D Function Trajectory', fontsize=18)
         

         
ax2 = plt.subplot(1, 3, 2)
plt.plot(samp_ts, x_, color='#BB5566', linewidth=2.2, label='True x')
plt.plot(samp_ts, x_vel_true, color='#004488', linewidth=2.2, label='True x vel')
plt.plot(samp_ts, aug, color='#DDAA33', linewidth=2.2, label='Learnt aug')
rc('font', family='serif')
rc('text', usetex=True)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.legend(loc='upper left', fontsize=14)
plt.title('ANODE(1) x vs t', fontsize=18)




ax3 = plt.subplot(1, 3, 3)
plt.plot(samp_ts, y_, color='#BB5566', linewidth=2.2, label='True y')
plt.plot(samp_ts, y_vel_true, color='#004488', linewidth=2.2, label='True y vel')
plt.plot(samp_ts, aug, color='#DDAA33', linewidth=2.2, label='Learnt aug')
rc('font', family='serif')
rc('text', usetex=True)
plt.xlabel('t', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend(loc='upper left', fontsize=14)
plt.title('ANODE(1) y vs t', fontsize=18)



plt.tight_layout()
plt.savefig('anode1_2dfunc.png', bbox_inches='tight')


