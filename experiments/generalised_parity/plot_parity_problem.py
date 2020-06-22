import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import seaborn as sns


colours = ['#DDAA33', '#BB5566', '#004488']
markers = ['^', 'o', 'D']
labels = ['NODE', 'ANODE(D)', 'SONODE']


results = np.load('results.npy')
stats = np.empty((3, 2, 10, 2))

for i in range(3):
    for j in range(2):
        for k in range(10):
            stats[i][j][k][0] = np.mean(results[i][j][k])
            stats[i][j][k][1] = np.std(results[i][j][k])

means = np.empty((3, 2, 10))
stds = np.empty((3, 2, 10))
for i in range(3):
    for j in range(2):
        for k in range(10):
            means[i][j][k] = np.log(stats[i][j][k][0])
            stds[i][j][k] = stats[i][j][k][1]/stats[i][j][k][0]


dim = np.arange(1, 11, 1)

def plot_train(x):
    plt.scatter(dim, means[x][0], color=colours[x], marker=markers[x],\
                s=40, label=labels[x])
    plt.errorbar(dim, means[x][0], yerr=stds[x][0], color=colours[x], linewidth=1,\
                 linestyle='None', capsize=4, capthick = 1)
    #sns.set_style('white')


def plot_test(x):
    plt.scatter(dim, means[x][1], color=colours[x], marker=markers[x],\
                s=40, label=labels[x])
    plt.errorbar(dim, means[x][1], yerr=stds[x][0], color=colours[x], linewidth=1,\
                 linestyle='None', capsize=4, capthick = 1)
    

sns.set_style('darkgrid')
rc('font', family='serif')
rc('text', usetex=True)
plt.figure()
plot_train(0)
plot_train(1)
plot_train(2)
plt.title('Train MSE', fontsize=24)
plt.ylabel('ln(MSE)', fontsize=19)
plt.xlabel('Dimension', fontsize=19)
plt.legend(fontsize=14)
plt.ylim(-15, 0)
plt.savefig('parity_problem_train_single.png', bbox_inches='tight')

sns.set_style('darkgrid')
plt.figure()
plot_test(0)
plot_test(1)
plot_test(2)


rc('font', family='serif')
rc('text', usetex=True)
plt.title('Test MSE', fontsize=24)
plt.ylabel('ln(MSE)', fontsize=19)
plt.xlabel('Dimension', fontsize=19)
plt.legend(fontsize=14)
plt.ylim(-15, 0)
plt.savefig('parity_problem_test_single.png', bbox_inches='tight')
