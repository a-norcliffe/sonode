# using scaled time so each data point is 1 time unit


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--npoints', type=int, default=5000)
parser.add_argument('--experiment_no', type=int, default=1)
args = parser.parse_args()


from torchdiffeq import odeint_adjoint as odeint




data = pd.read_csv('SNLS80mv.csv', header=0, nrows=args.npoints)

data = data.values.tolist()
data = np.asarray(data)
data = np.transpose(data)

v1_data = data[0]
v2_data = data[1]

v1_data = v1_data - np.full_like(v1_data, np.mean(v1_data))
v2_data = v2_data - np.full_like(v2_data, np.mean(v2_data))
rescaling = 100
v1_data = rescaling*v1_data
v2_data = rescaling*v2_data
v2_tensor =torch.tensor(v2_data).float()
v2_tensor = v2_tensor.reshape(args.npoints, 1)

del data
del v2_data

def v1_func(time):
    if (time > len(v1_data)-1) or (time < 0):
        return 0
    else:
        t1 = int(math.floor(time))
        delta = time - t1
        if delta == 0:
            return v1_data[t1]
        else:
            return v1_data[t1]+delta*(v1_data[t1+1]-v1_data[t1])


class init_vel(nn.Module):
    
    def __init__(self, dim):
        super(init_vel, self).__init__()
        self.fc = nn.Linear(dim, dim)
        
    def forward(self, x0):
        out = self.fc(x0)
        return torch.cat((x0, out))


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=False)
        self.fc = nn.Linear(4*dim, dim)
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        cutoff = int(len(z)/2)
        x = z[:cutoff]
        v = z[cutoff:]
        t_ = t.detach().numpy()[0]
        v1 = torch.tensor([v1_func(t_)]).float()
        z_ = torch.cat((x, 0.01*x**3, v, v1))
        out = self.fc(z_)
        return torch.cat((v, out))
    

class ODEBlock(nn.Module):

    def __init__(self, odefunc, integration_times, indices):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = integration_times
        self.indices = indices

    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_times, rtol=args.tol, atol=args.tol,\
                     method='dopri5')
        out = out.gather(1, self.indices)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
       
        

if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    filename = 'sonode./'+str(args.experiment_no)+'./'
    data_dim = 1
    dim = data_dim
    #dim does not equal data_dim for ANODEs where they are augmented with extra zeros
    
    # model
    # making time samples
    samp_ts_array = np.arange(args.npoints)
    samp_ts = torch.tensor(samp_ts_array).float()
    samp_ts = samp_ts.reshape(args.npoints, 1)
    
    del samp_ts_array
    
    z0 = v2_tensor[0].to(device)

    model = torch.load(filename+'model.pth')
    ids = torch.arange(data_dim)
    ids = ids.repeat(args.npoints, 1)


    y0 = model[0](z0)
    del z0
    pred_z = odeint(model[1].odefunc, y0, samp_ts)
    pred_z = pred_z.gather(1, ids)
    pred_z = pred_z.detach().numpy().reshape(args.npoints)
    np.save(filename+'v2_test.npy', pred_z)
    
    
