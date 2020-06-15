import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0.1)





#node
sns.set_style('dark')
rc('font', family='serif')
rc('text', usetex=True)
ax1 = plt.subplot(1, 3, 1)
ninner = 40
nouter = 80

film_data = np.load('figure_data./node_film_2d.npy')

a = len(film_data)
frames = []

for i in range(a):
    frames += [film_data[i][:ninner+nouter]]
    
intermediate = np.asarray(frames)

inner = []
outer = []

for i in range(a):
    inner += [intermediate[i][:ninner]]
    outer += [intermediate[i][ninner:]]
    
inner = np.asarray(inner)
outer = np.asarray(outer)


#make film image:

inner_pic = np.empty((ninner, a, 2))
outer_pic = np.empty((nouter, a, 2))

for i in range(ninner):
    for j in range(a):
        inner_pic[i][j] = inner[j][i]
           
for i in range(ninner):
    inner_pic_plot = np.transpose(inner_pic[i])
    plt.plot(inner_pic_plot[0], inner_pic_plot[1], color='#004488', linewidth=0.3)    
inner_start_frame = np.transpose(inner[0])
inner_end_frame = np.transpose(inner[len(inner)-1])
plt.scatter(inner_start_frame[0], inner_start_frame[1], color='#004488', s=15)
plt.scatter(inner_end_frame[0], inner_end_frame[1], color='#004488', s=15)


for i in range(nouter):
    for j in range(a):
        outer_pic[i][j] = outer[j][i]
        
for i in range(nouter):
    outer_pic_plot = np.transpose(outer_pic[i])
    plt.plot(outer_pic_plot[0], outer_pic_plot[1], color='#BB5566', linewidth=0.3)    
outer_start_frame = np.transpose(outer[0])
outer_end_frame = np.transpose(outer[len(inner)-1])
plt.scatter(outer_start_frame[0], outer_start_frame[1], color='#BB5566', s=15)
plt.scatter(outer_end_frame[0], outer_end_frame[1], color='#BB5566', s=15)
#plt.xlabel('x', fontsize=14)
#plt.ylabel('y',fontsize=14)
rc('font', family='serif')
rc('text', usetex=True)
plt.xticks([])
plt.yticks([])
plt.title('NODE', fontsize=24)













#anode
#Selects what frame to stop at, for representation
a = 6


ninner = 40
nouter = 80

film_data = np.load('figure_data./anode_film_(2+1)d.npy')

frames = []

for i in range(len(film_data)):
    frames += [film_data[i][:ninner+nouter]]
    
intermediate = np.asarray(frames)

inner = []
outer = []

for i in range(a):
    inner += [intermediate[i][:ninner]]
    outer += [intermediate[i][ninner:]]
    
inner = np.asarray(inner)
outer = np.asarray(outer)



#make film image:
sns.set_style('white')
ax2 = fig.add_subplot(132, projection='3d')

inner_pic = np.empty((ninner, a, 3))
outer_pic = np.empty((nouter, a, 3))

for i in range(ninner):
    for j in range(a):
        inner_pic[i][j] = inner[j][i]
           
for i in range(ninner):
    inner_pic_plot = np.transpose(inner_pic[i])
    ax2.plot(inner_pic_plot[0], inner_pic_plot[1], inner_pic_plot[2], color='#004488', linewidth=0.3)    
inner_start_frame = np.transpose(inner[0])
inner_end_frame = np.transpose(inner[len(inner)-1])
ax2.scatter(inner_start_frame[0], inner_start_frame[1], inner_start_frame[2], color='#004488', s=15)
ax2.scatter(inner_end_frame[0], inner_end_frame[1], inner_end_frame[2], color='#004488', s=15)


for i in range(nouter):
    for j in range(a):
        outer_pic[i][j] = outer[j][i]
        
for i in range(nouter):
    outer_pic_plot = np.transpose(outer_pic[i])
    plt.plot(outer_pic_plot[0], outer_pic_plot[1], outer_pic_plot[2], color='#BB5566', linewidth=0.3)    
outer_start_frame = np.transpose(outer[0])
outer_end_frame = np.transpose(outer[len(inner)-1])
ax2.scatter(outer_start_frame[0], outer_start_frame[1], outer_start_frame[2], color='#BB5566', s=15)
ax2.scatter(outer_end_frame[0], outer_end_frame[1], outer_end_frame[2], color='#BB5566', s=15)
ax2.grid(False)
#ax2.xaxis.pane.fill = False
#ax2.yaxis.pane.fill = False
#ax2.zaxis.pane.fill = False

#ax2.set_xlabel('x', fontsize=14)
#ax2.set_ylabel('y', fontsize=14)
#ax2.set_zlabel('z', fontsize=14)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
rc('font', family='serif')
rc('text', usetex=True)
ax2.set_title('ANODE(1)', fontsize=24, pad=27)












#sonode
sns.set_style('dark')
ax3 = plt.subplot(1, 3, 3)
film_data = np.load('figure_data./sonode_film_2d.npy')

#which frame to stop by
a = len(film_data)

frames = []

for i in range(a):
    frames += [film_data[i][:120]]
    
intermediate = np.asarray(frames)

inner = []
outer = []

for i in range(a):
    inner += [intermediate[i][:40]]
    outer += [intermediate[i][40:]]
    
inner = np.asarray(inner)
outer = np.asarray(outer)


#make film image:

inner_pic = np.empty((40, a, 2))
outer_pic = np.empty((80, a, 2))

for i in range(40):
    for j in range(a):
        inner_pic[i][j] = inner[j][i]
           
for i in range(40):
    inner_pic_plot = np.transpose(inner_pic[i])
    plt.plot(inner_pic_plot[0], inner_pic_plot[1], color='#004488', linewidth=0.3)    
inner_start_frame = np.transpose(inner[0])
inner_end_frame = np.transpose(inner[len(inner)-1])
plt.scatter(inner_start_frame[0], inner_start_frame[1], color='#004488', s=15)
plt.scatter(inner_end_frame[0], inner_end_frame[1], color='#004488', s=15)


for i in range(80):
    for j in range(a):
        outer_pic[i][j] = outer[j][i]
        
for i in range(80):
    outer_pic_plot = np.transpose(outer_pic[i])
    plt.plot(outer_pic_plot[0], outer_pic_plot[1], color='#BB5566', linewidth=0.3)    
outer_start_frame = np.transpose(outer[0])
outer_end_frame = np.transpose(outer[len(inner)-1])
plt.scatter(outer_start_frame[0], outer_start_frame[1], color='#BB5566', s=15)
plt.scatter(outer_end_frame[0], outer_end_frame[1], color='#BB5566', s=15)
#plt.xlabel('x', fontsize=14)
#plt.ylabel('y',fontsize=14)
plt.xticks([])
plt.yticks([])
rc('font', family='serif')
rc('text', usetex=True)
plt.title('SONODE', fontsize=24)


plt.tight_layout()
#sns.set_style('white')
plt.savefig('nested_n_spheres.png', bbox_inches='tight')




