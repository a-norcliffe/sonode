import time
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualise', type=eval, default=True)
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evenly_spread_timestamps', type=eval, default=True)
parser.add_argument('--noise', type=float, default=0.0) #0.0 gives no noise
parser.add_argument('--npoints', type=int, default=50)
parser.add_argument('--extra_dim', type=int, default=1)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class ODEfunc(nn.Module):

    def __init__(self, dim, nhidden):
        super(ODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=False)
        self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


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

    
def function(t):
    return torch.exp(0.1667*t)
    


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    if args.extra_dim == 0:
        filename = 'node./'
    else:
        filename = 'anode('+str(args.extra_dim)+')./'
    
    os.makedirs('./'+filename)
    data_dim = 1
    dim = data_dim + args.extra_dim
    #dim does not equal data_dim for ANODEs where they are augmented with extra zeros

    # model
    # making time samples
    t0, t1, t2, tN = 0, 3, 7, 10
    if args.evenly_spread_timestamps:
        samp_ts1 = torch.linspace(t0, t1, int(args.npoints/2)).float()
        samp_ts2 = torch.linspace(t2, tN, int(args.npoints/2)).float()
        samp_ts = torch.cat((samp_ts1, samp_ts2))
        samp_ts = torch.reshape(samp_ts, (args.npoints, 1)).to(device)

    else:
        intermediate_times1 = torch.tensor(np.random.rand(int(args.npoints/2)-2, data_dim)*(t1 - t0)).float()
        intermediate_times1, indices = torch.sort(intermediate_times1, dim=0)
        samp_ts1 = torch.cat((torch.tensor([[t0]]).float(), intermediate_times1))
        samp_ts1 = torch.cat((samp_ts1, torch.tensor([[t1]]).float()))
        
        intermediate_times2 = torch.tensor(np.random.rand(int(args.npoints/2)-2, data_dim)*(tN - t2)).float()
        intermediate_times2, indices = torch.sort(intermediate_times2, dim=0)
        intermediate_times2 = intermediate_times2 + torch.full_like(intermediate_times2, t2)
        samp_ts2 = torch.cat((torch.tensor([[t2]]).float(), intermediate_times2))
        samp_ts2 = torch.cat((samp_ts2, torch.tensor([[tN]]).float()))
        samp_ts= torch.cat((samp_ts1, samp_ts2)).to(device)
       
        
    # making sampled data to fit
    z = function(samp_ts) 
    if args.noise != 0.0:
        noise = z.data.new(z.size()).normal_(0, args.noise)
        measured_z = (z + noise).to(device)
    else:
        measured_z = z.to(device)   
    zeros = torch.zeros((args.extra_dim)).float()
    z0 = torch.cat((z[0], zeros)).to(device)
        
    nhidden = 20
    
    # make indices for getting position
    ids = torch.arange(data_dim)
    ids = ids.repeat(args.npoints, 1)
    
    feature_layers = [ODEBlock(ODEfunc(dim, nhidden), samp_ts, ids)]
    model = nn.Sequential(*feature_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    nfe_arr = np.empty(args.niters)

    # training
    start_time = time.time()
    for itr in range(1, args.niters + 1):
        feature_layers[0].nfe = 0
        optimizer.zero_grad()
        #forward in time and solve ode
        pred_z = model(z0).to(device)
        # compute loss
        loss = loss_func(pred_z, measured_z)
        loss.backward()
        optimizer.step()
        # make arrays
        itr_arr[itr-1] = itr
        loss_arr[itr-1] = loss
        nfe_arr[itr-1] = feature_layers[0].nfe
        
        print('Iter: {}, running MSE: {:.4f}'.format(itr, loss))

    end_time = time.time()
    print('\n')
    print('Training complete after {} iterations.'.format(itr))
    loss = loss.detach().numpy()
    print('Train MSE = ' +str(loss))
    print('NFE = ' +str(feature_layers[0].nfe))
    print('Total time = '+str(end_time-start_time))
    print('No. parameters = '+str(count_parameters(model)))
    
    np.save(filename+'itr_arr.npy', np.asarray(itr_arr))
    np.save(filename+'nfe_arr.npy', np.asarray(nfe_arr))
    np.save(filename+'loss_arr.npy', np.asarray(loss_arr))
    torch.save(model, filename+'model.pth')
    
    # test
    ntest = 10
    test_ts = torch.linspace(t1, t2, ntest).float()
    test_ts = torch.reshape(test_ts, (10, 1))
    test_z = function(test_ts)
    pred_z = odeint(feature_layers[0].odefunc, z0, samp_ts)
    z1 = pred_z[int(args.npoints/2)-1]
    pred_z = odeint(feature_layers[0].odefunc, z1, test_ts)
    ids = torch.arange(data_dim)
    ids = ids.repeat(ntest, 1)
    pred_x_test = pred_z.gather(1, ids)
    loss = loss_func(pred_x_test, test_z).detach().numpy()
    print('Test MSE = '+str(loss))
    
    
    if args.visualise:
        nvis = 70 
        ts = torch.linspace(t0, tN, nvis)
        z_real = function(ts)
        ts = ts.numpy()
        z_real = z_real.numpy()
        plt.plot(ts, z_real, 'r', label='Real')
        np.save(filename+'ts.npy', ts)
        np.save(filename+'z_real.npy', z_real)
       
        samp_ts = samp_ts.numpy()
        measured_z = measured_z.numpy()
        
        plt.scatter(samp_ts, measured_z, label='Sampled')
        np.save(filename+'samp_ts.npy', samp_ts)
        np.save(filename+'measured_z.npy', measured_z)
       
        test_ts = test_ts.numpy()
        test_ts = np.reshape(test_ts, (ntest))
        test_z = test_z.detach().numpy()
        test_z = np.reshape(test_z, (ntest))
        plt.scatter(test_ts, test_z, marker='x', color='k', label='Test')
        np.save(filename+'test_ts.npy', test_ts)
        np.save(filename+'test_z.npy', test_z)
        
       # make indices for getting position
        ids = torch.arange(data_dim)
        ids = ids.repeat(nvis, 1)
        learnt_ts = torch.linspace(t0, tN, nvis).float()
        pred_z = odeint(feature_layers[0].odefunc, z0, learnt_ts)
        pred_x = pred_z.gather(1, ids)
        learnt_ts = learnt_ts.detach().numpy()
        pred_x = pred_x.detach().numpy()
        plt.plot(learnt_ts, pred_x, 'g', label='Learnt')
        np.save(filename+'learnt_ts.npy', learnt_ts)
        np.save(filename+'learnt_trajectory.npy', pred_x)
        plt.legend()
        plt.savefig(filename+'vis.png')
       
        

        
    

        
      

        
    


