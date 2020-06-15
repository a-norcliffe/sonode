# using scaled time so each data point is 1 time unit
# there is a lot of potential for underflow because of the x^3 term, so many
# runs may be needed

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
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--npoints', type=int, default=1000)
parser.add_argument('--extra_dim', type=int, default=1)
parser.add_argument('--experiment_no', type=int, default=1)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint




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


class init_aug(nn.Module):
    
    def __init__(self, data_dim_, extra_dim_):
        super(init_aug, self).__init__()
        self.fc = nn.Linear(data_dim_, extra_dim_)
        
    def forward(self, x0):
        out = self.fc(x0)
        return torch.cat((x0, out))


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.fc = nn.Linear(2+dim, dim)
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        cutoff = data_dim
        x = z[:cutoff]
        a = z[cutoff:]
        t_ = t.detach().numpy()[0]
        v1 = torch.tensor([v1_func(t_)]).float()
        z_ = torch.cat((x, 0.01*x**3, a, v1))
        out = self.fc(z_)
        return out
    

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
       
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        



if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    filename = 'anode('+str(args.extra_dim)+')./'+str(args.experiment_no)+'./'
    data_dim = 1
    dim = data_dim + args.extra_dim
    #dim does not equal data_dim for ANODEs where they are augmented with extra zeros
    
    # model
    # making time samples
    samp_ts_array = np.arange(args.npoints)
    samp_ts = torch.tensor(samp_ts_array).float()
    samp_ts = samp_ts.reshape(args.npoints, 1)

    z0 = v2_tensor[0].to(device)    
    
    # make indices for getting position
    ids = torch.arange(data_dim)
    ids = ids.repeat(args.npoints, 1)
    
    feature_layers = [init_aug(data_dim, args.extra_dim), ODEBlock(ODEfunc(dim), samp_ts, ids)]
    model = nn.Sequential(*feature_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    nfe_arr = np.empty(args.niters)
    time_arr = np.empty(args.niters)

    # training
    start_time = time.time()
    # set arbitrary maximum acceptable loss
    min_loss = 1.0
    for itr in range(1, args.niters + 1):
        feature_layers[1].nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()
        #forward in time and solve ode
        pred_z = model(z0).to(device)
        # compute loss
        loss = loss_func(pred_z, v2_tensor)
        loss.backward()
        optimizer.step()
        # make arrays
        iter_end_time = time.time()
        itr_arr[itr-1] = itr
        loss_arr[itr-1] = loss
        nfe_arr[itr-1] = feature_layers[1].nfe
        time_arr[itr-1] = iter_end_time-iter_start_time
        
        print('Iter: {}, running MSE: {:.4f}'.format(itr, loss))
        if loss  < min_loss:
            min_loss = loss
            torch.save(model, filename+'model.pth')

    end_time = time.time()
    print('\n')
    print('Training complete after {} iterations.'.format(itr))
    loss = loss.detach().numpy()
    print('Train MSE = ' +str(loss))
    print('NFE = ' +str(feature_layers[1].nfe))
    print('Total time = '+str(end_time-start_time))
    print('No. parameters = '+str(count_parameters(model)))
    print('Minimum Loss = '+str(min_loss.detach().numpy()))

    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'nfe_arr.npy', nfe_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    np.save(filename+'time_arr.npy', time_arr)
    