import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--extra_dim', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class ODEfunc(nn.Module):

    def __init__(self, dim, nhidden):
        super(ODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
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

    def __init__(self, odefunc, t0_, tN_, indices):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = torch.tensor([t0_, tN_]).float()
        self.indices = indices
        
    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_times, rtol=args.tol, atol=args.tol)
        out = out[1].gather(1, self.indices)
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
    data_dim = 1
    dim = data_dim + args.extra_dim
    
    if args.extra_dim == 0:
        filename = 'node./'
    else:
        filename = 'anode('+str(args.extra_dim)+')./'
    
    try:
        os.makedirs('./'+filename)
    except FileExistsError:
        pass
    
   # making the data
    zeros = torch.zeros(2, args.extra_dim).float()
    z0 = torch.cat((torch.tensor([[1.0], [-1.0]]).float(), zeros), dim=1).to(device)
    zN = torch.tensor([[-1.0], [1.0]]).float().to(device)
    ntotal = 2

    # model
    t0, tN = 0, 1
    nhidden = 20
    
    # indices to unagument zN
    ids = torch.zeros((ntotal, 1)).long()
    
    feature_layers = [ODEBlock(ODEfunc(dim, nhidden), t0, tN, ids)]
    model = nn.Sequential(*feature_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    nfe_arr = np.empty(args.niters)
    time_arr = np.empty(args.niters)

    # training
    min_loss = 2.0
    start_time = time.time()
    for itr in range(1, args.niters + 1):
        model[0].nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()

        # forward in time and solve ode
        pred_z = model(z0)
        # compute loss
        loss = loss_func(pred_z, zN)
        loss.backward()
        optimizer.step()
        iter_end_time = time.time()
        #make arrays
        itr_arr[itr-1] = itr
        loss_arr[itr-1] = loss
        nfe_arr[itr-1] = model[0].nfe
        time_arr[itr-1] = iter_end_time-iter_start_time
        
        print('Iter: {}, running MSE: {:.4f}'.format(itr, loss))
                            
            
    end_time = time.time()
    print('\n')
    print('Training complete after {} iters.'.format(itr))
    print('Time = ' + str(end_time-start_time))
    loss = loss_func(pred_z, zN).detach().numpy()
    print('Train MSE = ' +str(loss))
    print('NFE = ' +str(feature_layers[0].nfe))
    print('Parameters = '+str(count_parameters(model)))
    
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'nfe_arr.npy', nfe_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    np.save(filename+'time_arr.npy', time_arr)
    torch.save(model, filename+'model.pth')

    if args.visualize:
        ntimestamps = 30
        ts = torch.tensor(np.linspace(t0, tN, ntimestamps)).float()
        pred_z = odeint(model[0].odefunc, z0, ts)
        pred_z = pred_z.detach().numpy()
        trajectory_1 = []
        trajectory_2 = []
        
        for i in range(ntimestamps):
            trajectory_1 += [pred_z[i][0][0]]
            trajectory_2 += [pred_z[i][1][0]]
            
        trajectory_1 = np.asarray(trajectory_1)
        trajectory_1 = np.reshape(trajectory_1, (ntimestamps))
        trajectory_2 = np.asarray(trajectory_2)
        trajectory_2 = np.reshape(trajectory_2, (ntimestamps))
        
        ts = ts.detach().numpy()
        plt.figure()
        plt.plot(ts, trajectory_1, 'blue')
        plt.plot(ts, trajectory_2, 'red')
        
        plt.savefig(filename+'vis.png')
        np.save(filename+'ts.npy', ts)
        np.save(filename+'trajectory_1.npy', trajectory_1)
        np.save(filename+'trajectory_2.npy', trajectory_2)
    


