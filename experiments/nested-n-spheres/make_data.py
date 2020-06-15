import numpy as np


npoints = 40
ninner = 20
nouter = 20
dim = 3
test_train = 'test' #choose train or test to choose the name
r = np.array([[0, 0.3, 0.7, 1], [0, 0.5, 0.85, 1], [0, 0.3, 0.7, 1], [0, 0.3, 0.7, 1]])
r_sq = r[dim-1]**2

name_in = 'data./'+str(dim)+'din_'+str(npoints)+'_'+test_train+'.npy'
name_out = 'data./'+str(dim)+'dout_'+str(npoints)+'_'+test_train+'.npy'

in_data = np.empty((npoints, dim))
out_data = np.empty((npoints, 1))

i = 0
while i < ninner:
    point = np.random.rand(dim)*2 - 1
    dot = np.dot(point, point)
    if (r_sq[0] <= dot) and (dot <= r_sq[1]):
        in_data[i] = point
        out_data[i] = [-1]
        i += 1
    else:
        pass
  
i = 0    
while i < nouter:   
    point = np.random.rand(dim)*2 - 1
    dot = np.dot(point, point)
    if (r_sq[2] <= dot) and (dot <= r_sq[3]):
        in_data[i+ninner] = point
        out_data[i+ninner] = [1]
        i += 1
    else:
        pass

 
np.save(name_in, in_data)
np.save(name_out, out_data)