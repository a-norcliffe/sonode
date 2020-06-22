"""
Test the double function learnt solution with a theoretical solution. Change
the coefficient C to find the best theoretical fit to the learnt augmented
solution. If the sign of C flips then the augmented trajectory will flip in
the t axis
"""

import matplotlib.pyplot as plt
from matplotlib import rc as rc
import seaborn as sns
import numpy as np
import scipy.integrate as integrate
import argparse

filename = 'results./double_func./not_mixed./'

parser = argparse.ArgumentParser()
parser.add_argument('--C', type=float, default=1)
args = parser.parse_args()

#solve the ode
omega = 1
gamma = 0.1667
times = np.linspace(0, 10, 50)

#change C to change the theoretical augmented trajectory
C = args.C

def derivatives(z, t):
    x = z[0]
    a = z[1]
    vx = C*a - omega*x - gamma*x + omega
    va = omega*a - gamma*a -(2*omega**2*x + gamma*omega - omega**2)/C
    return np.array([vx, va])

start = [0, 1]
colours = ['#004488', '#BB5566']
names = ['sin', 'cos']
def run_ode(x):
    z0 = [start[x], 0]
    z_arr = integrate.odeint(derivatives, z0, times)
    z_arr = np.transpose(z_arr)
    x_arr = z_arr[0]
    a_arr = z_arr[1]
    np.save(filename+names[x]+'_theory_x.npy', x_arr)
    np.save(filename+names[x]+'_theory_a.npy', a_arr)
  
run_ode(0)
run_ode(1)


# load data
theory_sin_z = np.load(filename+'sin_theory_x.npy')
theory_sin_a = np.load(filename+'sin_theory_a.npy')
theory_cos_z = np.load(filename+'cos_theory_x.npy')
theory_cos_a = np.load(filename+'cos_theory_a.npy')

learnt_f1 = np.load(filename+'learnt_f1.npy')
learnt_f2 = np.load(filename+'learnt_f2.npy')
samp_ts = np.load(filename+'ts.npy')
real_f1 = np.load(filename+'real_f1.npy')
real_f2 = np.load(filename+'real_f2.npy')
learnt_a1 = np.load(filename+'learnt_a1.npy')
learnt_a2 = np.load(filename+'learnt_a2.npy')


#plot figure
sns.set_style('dark')
rc('font', family='serif')
rc('text', usetex=True)
plt.figure()
rc('font', family='serif')
rc('text', usetex=True)
plt.plot(samp_ts, learnt_f1, color='#004488', label='Learnt $x_{1}$', linewidth=2.2)
plt.scatter(samp_ts, real_f1, color='#004488', s=25)
plt.plot(samp_ts, theory_sin_a, color='#DDAA33', label='Theory $a_{1}$', linewidth=2.2)
plt.plot(samp_ts, learnt_a1, color = '#004488', label='Learnt $a_{1}$', linestyle='--', linewidth=2.3)
plt.plot(samp_ts, learnt_f2, color='#BB5566', label='Learnt $x_{2}$', linewidth=2.2)
plt.scatter(samp_ts, real_f2, color='#BB5566', s=25)
plt.plot(samp_ts, theory_cos_a, color='#DDAA33', label='Theory $a_{2}$', linewidth=2.2)  
plt.plot(samp_ts, learnt_a2, color='#BB5566', label='Learnt $a_{2}$', linestyle='--', linewidth=2.3)     
plt.xlabel('t', fontsize=18)
plt.ylabel('$x_{1}, x_{2}, a_{1}, a_{2}$', fontsize=18)
plt.title('ANODE(1) Double Function C = '+str(args.C), fontsize=20)
plt.legend(ncol=2, fontsize=12)
plt.savefig('not_mixed_double_func_with_theory_a.png')
