import time
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
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--ntrainpoints', type=int, default=70)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


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
    full_ts = torch.load('data./time_data.pt')
    
    train_z = full_z[:args.ntrainpoints]
    train_ts = full_ts[:args.ntrainpoints]
    test_z = full_z[args.ntrainpoints:]
    
    
    z0 = train_z[0].to(device)
    
    # make indices for getting position
    ids = torch.arange(data_dim)
    ids = ids.repeat(args.ntrainpoints, 1)
    
    feature_layers = [initial_velocity(dim), ODEBlock(ODEfunc(dim), train_ts, ids)]
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
    min_loss = 10
    for itr in range(1, args.niters + 1):
        model[1].nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()
        #forward in time and solve ode
        pred_z = model(z0).to(device)
        # compute loss
        loss = loss_func(pred_z, train_z)
        loss.backward()
        optimizer.step()
        iter_end_time = time.time()
        # make arrays
        itr_arr[itr-1] = itr
        loss_arr[itr-1] = loss
        nfe_arr[itr-1] = model[1].nfe
        time_arr[itr-1] = iter_end_time-iter_start_time
        
        print('Iter: {}, running MSE: {:.4f}'.format(itr, loss))
        
        if loss  < min_loss:
            min_loss = loss
            torch.save(model, filename+'./model.pth')
            

    end_time = time.time()
    
    results = np.load('results.npy')
    
    print('\n')
    print('Training complete after {} iterations.'.format(itr))
    loss = min_loss.detach().numpy()
    results[0][0][int(args.experiment_no-1)] = loss
    print('Train MSE = ' +str(loss))
    print('NFE = ' +str(model[1].nfe))
    print('Total time = '+str(end_time-start_time))
    print('No. parameters = '+str(count_parameters(model)))
    
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'nfe_arr.npy', nfe_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    np.save(filename+'time_arr.npy' , time_arr)

    
    # test
    model = torch.load(filename+'model.pth')
    y0 = model[0](z0)
    pred_z = odeint(model[1].odefunc, y0, full_ts)
    ids = torch.arange(data_dim)
    ids = ids.repeat(len(full_ts), 1)
    pred_z_test = pred_z.gather(1, ids)
    pred_x_test = pred_z_test[args.ntrainpoints:]
    test_loss = loss_func(pred_x_test, test_z).detach().numpy()
    results[1][0][int(args.experiment_no-1)] = test_loss
    print('Test MSE = '+str(test_loss))
    
    np.save('results.npy', results)
    
    if args.visualise:
        ts_array = full_ts.detach().numpy().reshape(len(full_ts))
        real_z = full_z.detach().numpy().reshape(len(ts_array))
        pred_z_test = pred_z_test.detach().numpy().reshape(len(ts_array))
        plt.plot (ts_array, real_z, color='b', label='True')
        plt.plot(ts_array, pred_z_test, color= 'r', label='Learnt')
        np.save(filename+'learnt_trajectory.npy', pred_z_test)
        plt.axvline(x=ts_array[args.ntrainpoints], linestyle='--', color='k')
        plt.legend(loc='upper left')
        plt.ylabel('x')
        plt.xlabel('t')
        plt.title('SONODE Experiment = '+str(args.experiment_no))
        plt.savefig(filename+'vis.png')
