import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns


fig = plt.figure(figsize=[10, 4])
fig.subplots_adjust(hspace=0., wspace=0)


sns.set_style('dark')
rc('font', family='serif')
rc('text', usetex=True)
ax1 = plt.subplot(1, 2, 1)
filename = 'results./double_func./mixed./'
learnt_f1 = np.load(filename+'learnt_f1.npy')
learnt_f2 = np.load(filename+'learnt_f2.npy')
samp_ts = np.load(filename+'ts.npy')
real_f1 = np.load(filename+'real_f1.npy')
real_f2 = np.load(filename+'real_f2.npy')
learnt_a1 = np.load(filename+'learnt_a1.npy')
learnt_a2 = np.load(filename+'learnt_a2.npy')

plt.plot(samp_ts, learnt_f1, color='#004488', label='Learnt $x_{1}$')
plt.plot(samp_ts, learnt_f2, color='#BB5566', label='Learnt $x_{2}$')
plt.scatter(samp_ts, real_f1, color='#004488', s=20)#, label='True x_1')
plt.scatter(samp_ts, real_f2, color='#BB5566', s=20)#, label='True x_2')
plt.plot(samp_ts, learnt_a1, color = '#004488', label='$a_{1}$', linestyle='--')
plt.plot(samp_ts, learnt_a2, color='#BB5566', label='$a_{2}$', linestyle='--')
        #plt.legend()
plt.xlabel('t', fontsize=18)
plt.ylabel('$x_{1}, x_{2}, a_{1}, a_{2}$', fontsize=18)
plt.title('ANODE(1) Double Function', fontsize=20)
#sns.set_style('white')
plt.legend(loc='upper right', ncol=2, fontsize=13)



sns.set_style('dark')
ax2 = plt.subplot(1, 2, 2)

filename = 'results./triple_func./'
learnt_f1 = np.load(filename+'learnt_f1.npy')
learnt_f2 = np.load(filename+'learnt_f2.npy')
learnt_f3 = np.load(filename+'learnt_f3.npy')
samp_ts = np.load(filename+'ts.npy')
real_f1 = np.load(filename+'real_f1.npy')
real_f2 = np.load(filename+'real_f2.npy')
real_f3 = np.load(filename+'real_f3.npy')
learnt_a1 = np.load(filename+'learnt_a1.npy')
learnt_a2 = np.load(filename+'learnt_a2.npy')
learnt_a3 = np.load(filename+'learnt_a3.npy')

rc('font', family='serif')
rc('text', usetex=True)
plt.plot(samp_ts, learnt_f1, color='#004488', label='Learnt $x_{1}$')
plt.plot(samp_ts, learnt_f2, color='#BB5566', label='Learnt $x_{2}$')
plt.plot(samp_ts, learnt_f3, color='#DDAA33', label='Learnt $x_{3}$')
plt.scatter(samp_ts, real_f1, color='#004488', s=20)#, label='True x_1')
plt.scatter(samp_ts, real_f2, color='#BB5566', s=20)#, label='True x_2')
plt.scatter(samp_ts, real_f3, color='#DDAA33', s=20)#, label='True x_3')
plt.plot(samp_ts, learnt_a1, color = '#004488', label='$a_{1}$', linestyle='--')
plt.plot(samp_ts, learnt_a2, color='#BB5566', label='$a_{2}$', linestyle='--')
plt.plot(samp_ts, learnt_a3, color='#DDAA33', label='$a_{3}$', linestyle='--')
        #plt.legend()
plt.xlabel('t', fontsize=18)
plt.ylabel('$x_{1}, x_{2}, x_{3}, a_{1}, a_{2}, a_{3}$', fontsize=18)
plt.title('ANODE(1) Triple Function', fontsize=20)
#sns.set_style('white')
plt.legend(ncol=2, fontsize=13)


plt.tight_layout()
plt.savefig('triple_func.png', bbox_inches='tight')