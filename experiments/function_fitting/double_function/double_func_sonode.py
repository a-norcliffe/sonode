import time
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
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--ntimestamps',type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class initial_velocity(nn.Module):
    
    def __init__(self, dim, nhidden):
        super(initial_velocity, self).__init__()
        self.tanh = nn.Hardtanh(min_val=-5, max_val=5, inplace=False)
        self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        
    def forward(self, x0):
        out = self.fc1(x0)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return torch.cat((x0, out))


class ODEfunc(nn.Module):

    def __init__(self, dim, nhidden):
        super(ODEfunc, self).__init__()
        self.elu = nn.ELU(alpha=1, inplace=False)
        self.fc1 = nn.Linear(2*dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.nfe = 0

    def forward(self, t, x):
        cutoff = int(len(x)/2)
        z = x[:cutoff]
        v = x[cutoff:]
        into = torch.cat((z, v), dim=1)
        self.nfe += 1
        out = self.fc1(into)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return torch.cat((v, out))
    

class ODEBlock(nn.Module):

    def __init__(self, odefunc, times):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = times
        
    def forward(self, x):
        integrated = odeint(self.odefunc, x, self.integration_times, rtol=args.tol, atol=args.tol)
        out = torch.empty(args.ntimestamps, 2, 1)
        for i in range(args.ntimestamps):
            out[i] = integrated[i][:2]
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



a, b, c, d = 1, -3, 2, 1

def f1(t):
    return (a*torch.sin(t)+b*torch.cos(t))*torch.exp(-0.1667*t)


def f2(t):
    return (c*torch.sin(t)+d*torch.cos(t))*torch.exp(-0.1667*t)



if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    filename = 'results./double_func./sonode./'
    dim = 1
    
    # times
    t0, tN = 0, 10
    samp_ts = torch.linspace(t0, tN, args.ntimestamps)
    
    # make data
    z0= torch.tensor([[f1(samp_ts[0])], [f2(samp_ts[0])]]).float().to(device)
    z = torch.empty((args.ntimestamps, 2, 1)).float()
    for i in range(args.ntimestamps):
        z[i] = torch.tensor([[f1(samp_ts[i]).float()],\
         [f2(samp_ts[i]).float()]])
    
    #model
    nhidden = 20
    feature_layers = [initial_velocity(dim, nhidden), ODEBlock(ODEfunc(dim, nhidden), samp_ts)]
    model = nn.Sequential(*feature_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    nfe_arr = np.empty(args.niters)
    time_arr = np.empty(args.niters)

    #training
    start_time = time.time()
    for itr in range(1, args.niters + 1):
        feature_layers[1].nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()

        # forward in time and solve ode
        pred_z = model(z0)
        # compute loss
        loss = loss_func(pred_z, z)
        loss.backward()
        optimizer.step()
        iter_end_time = time.time()
        #make arrays
        itr_arr[itr-1] = itr
        loss_arr[itr-1] = loss
        nfe_arr[itr-1] = feature_layers[1].nfe
        time_arr[itr-1] = iter_end_time-iter_start_time
        
        print('Iter: {}, running MSE: {:.4f}'.format(itr, loss))


    end_time = time.time()
    print('\n')
    print('Training complete after {} iters.'.format(itr))
    print('Time = ' + str(end_time-start_time))
    loss = loss_func(pred_z, z).detach().numpy()
    print('Train MSE = ' +str(loss))
    print('NFE = ' +str(feature_layers[1].nfe))
    print('Parameters = '+str(count_parameters(model)))
    
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'nfe_arr.npy', nfe_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    torch.save(model, filename+'model.pth')
    
    
    if args.visualize:
        pred_z = pred_z.detach().numpy()
        
        ids = torch.full((2, 1), 1).long()
        y0 = feature_layers[0](z0)
        integrated = odeint(feature_layers[1].odefunc, y0, samp_ts)
        pred_a = torch.empty(args.ntimestamps, 2, 1)
        for i in range(args.ntimestamps):
            pred_a[i] = integrated[i][2:]
            
        real_f1 = f1(samp_ts).numpy()
        real_f2 = f2(samp_ts).numpy()
        samp_ts = samp_ts.numpy()
        plt.plot(samp_ts, real_f1, color='b')
        plt.plot(samp_ts, real_f2, color='r')
        plt.scatter(samp_ts, real_f1, color='b')
        plt.scatter(samp_ts, real_f2, color='r')
        learnt_f1 = np.empty((args.ntimestamps))
        learnt_f2 = np.empty((args.ntimestamps))
        for i in range(args.ntimestamps):
            learnt_f1[i] = pred_z[i][0]
            learnt_f2[i] = pred_z[i][1]
        plt.plot(samp_ts, learnt_f1)
        plt.plot(samp_ts, learnt_f2)
        plt.savefig(filename+'real.png')

        plt.figure()
        
        plt.plot(samp_ts, real_f1, color='b')
        plt.plot(samp_ts, real_f2, color='r')
        plt.scatter(samp_ts, real_f1, color='b')
        plt.scatter(samp_ts, real_f2, color='r')
        learnt_a1 = np.empty((args.ntimestamps))
        learnt_a2 = np.empty((args.ntimestamps))
        for i in range(args.ntimestamps):
            learnt_a1[i] = pred_a[i][0]
            learnt_a2[i] = pred_a[i][1]
        plt.plot(samp_ts, learnt_a1)
        plt.plot(samp_ts, learnt_a2)
        plt.savefig(filename+'vel.png')        
      

        
    


