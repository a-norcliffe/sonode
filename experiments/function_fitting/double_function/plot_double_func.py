import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

filename = 'results./double_func./not_mixed./'
learnt_f1 = np.load(filename+'learnt_f1.npy')
learnt_f2 = np.load(filename+'learnt_f2.npy')
samp_ts = np.load(filename+'ts.npy')
real_f1 = np.load(filename+'real_f1.npy')
real_f2 = np.load(filename+'real_f2.npy')
learnt_a1 = np.load(filename+'learnt_a1.npy')
learnt_a2 = np.load(filename+'learnt_a2.npy')

sns.set_style('dark')
rc('font', family='serif')
rc('text', usetex=True)
plt.figure()
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
plt.legend(ncol=2, fontsize=13)
plt.savefig('not_mixed_double_func.png', bbox_inches='tight')