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
parser.add_argument('--niters', type=int, default=2200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ntimestamps', type=int, default=100)
parser.add_argument('--extra_dim', type=int, default=2)
parser.add_argument('--experiment_no', type=int, default=1)
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
        integrated = odeint(self.odefunc, x, self.integration_times, rtol=args.tol, atol=args.tol)
        out = integrated.gather(1, self.indices)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
       
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


gamma = np.array([0.1, 0.3])
omega = np.array([1, 1.2])

A, B, C, D = 1, 3, -5, 2
A_bar = np.array([omega[0]*B-gamma[0]*A, omega[1]*D-gamma[1]*C])
B_bar = np.array([gamma[0]*B+omega[0]*A, gamma[1]*D+omega[1]*C])

def x(t):
    return torch.exp(-gamma[0]*t)*(A*torch.cos(omega[0]*t)+B*torch.sin(omega[0]*t))
 
    
def real_vel_x(t):
    return torch.exp(-gamma[0]*t)*((A_bar[0])*torch.cos(omega[0]*t)-(B_bar[0])*torch.sin(omega[0]*t))


def y(t):
    return torch.exp(-gamma[1]*t)*(C*torch.cos(omega[1]*t)+D*torch.sin(omega[1]*t))


def real_vel_y(t):
    return torch.exp(-gamma[1]*t)*((A_bar[1])*torch.cos(omega[1]*t)-(B_bar[1])*torch.sin(omega[1]*t))




if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    filename = 'results./anode('+str(args.extra_dim)+')./start_at_zero./'+str(args.experiment_no)+'./'
    os.makedirs('./'+filename)
    data_dim = 2
    dim = data_dim + args.extra_dim
    #dim does not equal data_dim for ANODEs where they are augmented with extra zeros
    
    # model
    # making time samples
    t0, tN = 0, 10

    samp_ts = torch.linspace(t0, tN, args.ntimestamps).float()
    samp_ts = torch.reshape(samp_ts, (args.ntimestamps, 1)).to(device)
    
   # making sampled data to fit
    x_ = x(samp_ts)
    y_ = y(samp_ts)
       
    z = torch.cat((x_, y_), dim=1).float().to(device)
    z0 = z[0].float().to(device)
    
    zeros = torch.zeros(args.extra_dim)
    z0 = torch.cat((z0, zeros))
    
    nhidden = 10
    
    ids = torch.arange(data_dim)
    ids = ids.repeat(args.ntimestamps, 1).long()
    
    feature_layers = [ODEBlock(ODEfunc(dim, nhidden), samp_ts, ids)]
    model = nn.Sequential(*feature_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    nfe_arr = np.empty(args.niters)
    time_arr = np.empty(args.niters)

    # training
    start_time = time.time()
    for itr in range(1, args.niters + 1):
        feature_layers[0].nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()
        #forward in time and solve ode
        pred_z = model(z0).to(device)
        # compute loss
        loss = loss_func(pred_z, z)
        loss.backward()
        optimizer.step()
        # make arrays
        iter_end_time = time.time()
        itr_arr[itr-1] = itr
        loss_arr[itr-1] = loss
        nfe_arr[itr-1] = feature_layers[0].nfe
        time_arr[itr-1] = iter_end_time-iter_start_time
        
        print('Iter: {}, running MSE: {:.4f}'.format(itr, loss))

    end_time = time.time()
    print('\n')
    print('Training complete after {} iterations.'.format(itr))
    loss = loss.detach().numpy()
    print('Train MSE = ' +str(loss))
    print('NFE = ' +str(feature_layers[0].nfe))
    print('Total time = '+str(end_time-start_time))
    print('No. parameters = '+str(count_parameters(model)))
    
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'nfe_arr.npy', nfe_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    np.save(filename+'time_arr.npy', time_arr)
    torch.save(model, filename+'model.pth')
    
    x_ = x_.numpy()
    x_ = np.transpose(x_)[0]

    y_ = y_.numpy()
    y_ = np.transpose(y_)[0]
    
    x_vel_true = real_vel_x(samp_ts).numpy()
    x_vel_true = np.transpose(x_vel_true)[0]
    
    y_vel_true = real_vel_y(samp_ts).numpy()
    y_vel_true = np.transpose(y_vel_true)[0]
    
    to_plot = pred_z.reshape((args.ntimestamps, data_dim))
    to_plot = to_plot.detach().numpy()
    to_plot = np.transpose(to_plot)
    x_pos = to_plot[0]
    y_pos = to_plot[1]
    plt.plot(x_pos, y_pos, label='Learnt', color='r')
    plt.plot(x_, y_, label='True', color='b')
    plt.legend()
    plt.savefig(filename+'trajectories.png')
    
    plot_ts = samp_ts.detach().numpy()
    plot_ts = np.transpose(plot_ts)[0]
    
    plt.figure()
    plt.plot(plot_ts, x_, label='True x', color='b')
    plt.plot(plot_ts, x_pos, label='Learnt x', color='r')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.legend()
    plt.savefig(filename+'x.png')
    
    plt.figure()
    plt.plot(plot_ts, y_, label='True y', color='b')
    plt.plot(plot_ts, y_pos, label='Learnt y', color='r')
    plt.ylabel('y')
    plt.xlabel('t')
    plt.legend()
    plt.savefig(filename+'y.png')
    
    if args.extra_dim == 1:
        integrated = odeint(feature_layers[0].odefunc, z0, samp_ts)
        ids = torch.full((args.ntimestamps, 1), 2).long()
        aug = integrated.gather(1, ids)
        
        aug = aug.detach().numpy()
        
        plt.figure()
        plt.plot(samp_ts, aug, label='aug', color='g')
        plt.plot(samp_ts, x_, label='x', color ='b')
        plt.plot(samp_ts, y_, label='y', color='r')
        plt.legend()
        plt.savefig(filename+'aug.png')
        np.save(filename+'learnt_x.npy', x_pos)
        np.save(filename+'learnt_y.npy', y_pos)
        np.save(filename+'learnt_aug.npy', aug)
        
    elif args.extra_dim == 2:
        integrated = odeint(feature_layers[0].odefunc, z0, samp_ts)
        ids1 = torch.full((args.ntimestamps, 1), 2).long()
        ids2 = torch.full((args.ntimestamps, 1), 3).long()
        ids = torch.cat((ids1, ids2), dim=1)
        aug = integrated.gather(1, ids)
        
        aug = aug.detach().numpy()
        aug = np.transpose(aug)
        aug_x = aug[0]
        aug_y = aug[1]
        
        plt.figure()
        plt.plot(samp_ts, aug_x, label='aug x', color='g')
        plt.plot(plot_ts, x_, label='x', color ='b')
        plt.plot(plot_ts, y_, label='y', color='r')
        plt.legend()
        plt.savefig(filename+'aug_x.png')
        
        plt.figure()
        plt.plot(samp_ts, aug_y, label='aug y', color='g')
        plt.plot(plot_ts, x_, label='x', color ='b')
        plt.plot(plot_ts, y_, label='y', color='r')
        plt.legend()
        plt.savefig(filename+'aug_y.png')
       
        plt.figure()
        plt.plot(x_, y_, label='True func', color='b')
        plt.plot(x_pos, y_pos, label='Learnt func', color='r')
        plt.plot(x_vel_true, y_vel_true, label='True vel', color = 'k')
        plt.plot(aug_x, aug_y, label='aug', color='g')
        plt.legend(loc='upper left')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(filename+'aug.png')
        
    else:
        pass
        

        
    

        
      

        
    


