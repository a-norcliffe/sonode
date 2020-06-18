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
parser.add_argument('--noise', type=float, default=0.7) #0.0 gives no noise
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--npoints', type=int, default=50)
parser.add_argument('--extrap', type=float, default=5.0)
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
    return torch.sin(t)
    


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    filename = 'results./'+str(args.noise)+'./experiment'+str(args.experiment_no)+'./anode('+str(args.extra_dim)+')./'
    os.makedirs('./'+filename)
    data_dim = 1
    dim = data_dim + args.extra_dim
    #dim does not equal data_dim for ANODEs where they are augmented with extra zeros

    # model
    # making time samples
    t0, tN = 0, 10
    if args.evenly_spread_timestamps:
        samp_ts = torch.linspace(t0, tN, args.npoints).float()
        samp_ts = torch.reshape(samp_ts, (args.npoints, 1))
    else:
        intermediate_times = torch.tensor(np.random.rand(args.npoints - 2, data_dim)*(tN - t0)).float()
        intermediate_times, indices = torch.sort(intermediate_times, dim=0)
        samp_ts = torch.cat((torch.tensor([[t0]]).float(), intermediate_times))
        samp_ts = torch.cat((samp_ts, torch.tensor([[tN]]).float()))
        
    # making sampled data to fit
    z = function(samp_ts) 

    measured_z = torch.load('data./noisy_sine_data_experiment_'+str(args.experiment_no)\
               +'_noise_'+str(args.noise)+'.pt').to(device)
    z0 = z[0].to(device)   
    
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
    
    results = np.load('results./results.npy')
    
    print('\n')
    print('Training complete after {} iterations.'.format(itr))
    loss = loss.detach().numpy()
    results[0][1][int(10*args.noise)][int(args.experiment_no-1)] = loss
    print('Train MSE = ' +str(loss))
    print('NFE = ' +str(feature_layers[0].nfe))
    print('Total time = '+str(end_time-start_time))
    print('No. parameters = '+str(count_parameters(model)))
    
    
    np.save(filename+'itr_arr.npy', np.asarray(itr_arr))
    np.save(filename+'nfe_arr.npy', np.asarray(nfe_arr))
    np.save(filename+'loss_arr.npy', np.asarray(loss_arr))
    torch.save(model, filename+'model.pth')

    
    # test
    extrap = args.extrap
    ntest = 10
    test_ts = torch.linspace(tN, tN+extrap, ntest).float()
    test_ts = torch.reshape(test_ts, (ntest, 1))
    test_z = function(test_ts)
    zN = odeint(feature_layers[0].odefunc, z0, torch.tensor([t0, tN]).float())
    zN = zN[1]
    pred_z = odeint(feature_layers[0].odefunc, zN, test_ts)
    ids = torch.arange(data_dim)
    ids = ids.repeat(ntest, 1)
    pred_x_test = pred_z.gather(1, ids)
    test_loss = loss_func(pred_x_test, test_z).detach().numpy()
    results[1][1][int(10*args.noise)][int(args.experiment_no-1)] = test_loss
    print('Test MSE = '+str(test_loss))
    
    np.save('results./results.npy', results)
    
    if args.visualise:
        nvis = 500 
        ts = torch.linspace(t0, tN+extrap, nvis)
        z_real = function(ts)
        ts = ts.numpy()
        z_real = z_real.numpy()
        plt.plot(ts, z_real, color='b', label='Real')
       
        samp_ts = samp_ts.numpy()
        measured_z = measured_z.numpy()
        
        plt.scatter(samp_ts, measured_z, label='Sampled', color='k')
       
        test_ts = test_ts.numpy()
        test_ts = np.reshape(test_ts, (ntest))
        test_z = test_z.detach().numpy()
        test_z = np.reshape(test_z, (ntest))
        plt.scatter(test_ts, test_z, marker='x', color='k', label='Test')
        
       # make indices for getting position
        ids = torch.arange(data_dim)
        ids = ids.repeat(nvis, 1)
        learnt_ts = torch.linspace(t0, tN+extrap, nvis).float()
        pred_z = odeint(feature_layers[0].odefunc, z0, learnt_ts)
        pred_x = pred_z.gather(1, ids)
        learnt_ts = learnt_ts.detach().numpy()
        pred_x = pred_x.detach().numpy()
        plt.plot(learnt_ts, pred_x, color='r', label='Learnt')
        np.save(filename+'learnt_trajectory.npy', pred_x)
        plt.legend(loc='lower left')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('ANODE('+str(args.extra_dim)+') Noise = '+str(args.noise)+' Experiment = '+str(args.experiment_no))
        plt.savefig(filename+'vis.png')
       
        

        
    

        
      

        
    


