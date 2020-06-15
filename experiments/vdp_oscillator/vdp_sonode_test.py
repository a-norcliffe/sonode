import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--experiment_no', type=int, default=1)
args = parser.parse_args()


from torchdiffeq import odeint_adjoint as odeint


omega = np.pi/5
def c(t):
    return torch.cos(omega*t)


class initial_velocity(nn.Module):
    
    def __init__(self, dim):
        super(initial_velocity, self).__init__()
        self.fc = nn.Linear(dim, dim)
        
    def forward(self, x0):
        out = self.fc(x0)
        return torch.cat((x0, out))


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.fc = nn.Linear(3*dim, dim)
        self.nfe = 0

    def forward(self, t, z):
        cutoff = int(len(z)/2)
        x = z[:cutoff]
        v = z[cutoff:]
        self.nfe += 1
        c_ = torch.tensor([c(t)]).float()
        z_ = torch.cat((x, v, c_))
        out = self.fc(z_)
        return torch.cat((v, out))
    
    
class ODEBlock(nn.Module):

    def __init__(self, odefunc, integration_times, indices):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = integration_times
        self.indices = indices

    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_times, rtol=args.tol, atol=args.tol)
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
    filename = 'sonode./'+str(args.experiment_no)+'./'
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_dim = 1
    dim = data_dim
    #dim does not equal data_dim for ANODEs where they are augmented with extra zeros

    # model
    
        
    # making sampled data to fit
    full_z = torch.load('data./position_data.pt')
    full_ts = torch.load('data./test_time_data.pt')

    
    
    z0 = full_z[0].to(device)
        
    # test
    model = torch.load(filename+'model.pth')
    y0 = model[0](z0)
    pred_z = odeint(model[1].odefunc, y0, full_ts)
    ids = torch.arange(data_dim)
    ids = ids.repeat(len(full_ts), 1)
    pred_z_test = pred_z.gather(1, ids)

    pred_z_test = pred_z_test.detach().numpy().reshape(len(full_ts))
    np.save(filename+'learnt_trajectory.npy', pred_z_test)
    

    
    

