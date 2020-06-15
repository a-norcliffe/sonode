import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualise', type=eval, default=True)
parser.add_argument('--niters', type=int, default=600)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--extra_dim', type=int, default=1)
parser.add_argument('--data_dimension', type=int, default=2)
parser.add_argument('--npoints', type=int, default=50)
parser.add_argument('--ntest', type=int, default=10)
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

    def __init__(self, odefunc, t0_, tN_):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = torch.tensor([t0_, tN_]).float()
        
    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_times, rtol=args.tol, atol=args.tol)
        out = out[1]
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        

class Decoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.tanh = nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, z):
        out = self.fc(z)
        out = self.tanh(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.extra_dim == 0:
        filename = 'node./'
    else:
        filename = 'anode('+str(args.extra_dim)+')./'
    
    dim = args.data_dimension + args.extra_dim
    
    #Download data
    name_in = str(args.data_dimension)+'din_'+str(args.npoints)+'_train.npy'
    name_out = str(args.data_dimension)+'dout_'+str(args.npoints)+'_train.npy'
    folder_name = 'data\\'
    z0 = torch.tensor(np.load(folder_name+name_in)).float().to(device)
    zN = torch.tensor(np.load(folder_name+name_out)).float().to(device)
    
    #Augment z0
    zeros = torch.zeros(args.npoints, args.extra_dim).float()
    z0 = torch.cat((z0, zeros), dim=1).to(device)

    # model
    t0, tN = 0, 1
    nhidden = 20
    feature_layers = [ODEBlock(ODEfunc(dim, nhidden), t0, tN), Decoder(dim, 1)]
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
        nfe_arr[itr-1] = feature_layers[0].nfe
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
    
    
    # make test data
    name_in = str(args.data_dimension)+'din_'+str(args.ntest)+'_test.npy'
    name_out = str(args.data_dimension)+'dout_'+str(args.ntest)+'_test.npy'
    folder_name = 'data./'
    z0 = torch.tensor(np.load(folder_name+name_in)).float().to(device)
    zN = torch.tensor(np.load(folder_name+name_out)).float().to(device)
    # augment z0
    zeros = torch.zeros(args.ntest, args.extra_dim).float()
    z0 = torch.cat((z0, zeros), dim=1).to(device)
    
    # run test data through network
    pred_z = model(z0)

    # compute loss
    loss = loss_func(pred_z, zN).detach().numpy()
    print('Test MSE = ' +str(loss))
    
    
    if args.visualise:
        samp_ts = torch.linspace(t0, tN, 30)
        if args.data_dimension == 1:
            z0 = torch.tensor(np.load('data./vis_data./1d_vis_data.npy')).float()
            ntotal = len(z0)
            #Augment z0
            zeros = torch.zeros(ntotal, args.extra_dim).float()
            z0 = torch.cat((z0, zeros), dim=1).to(device)
            pred_z = odeint(feature_layers[0].odefunc, z0, samp_ts)
            pred_z = pred_z.detach().numpy()
            if args.extra_dim == 0:
                name = 'figure_data./node_film_1d.npy'
            else:
                name = 'figure_data./anode_film_(1+'+str(args.extra_dim)+')d.npy'
            np.save(name, pred_z)
        elif args.data_dimension == 2:
            z0 = torch.tensor(np.load('data./vis_data./2d_vis_data.npy')).float()
            ntotal = len(z0)
            #Augment z0
            zeros = torch.zeros(ntotal, args.extra_dim).float()
            z0 = torch.cat((z0, zeros), dim=1).to(device)
            pred_z = odeint(feature_layers[0].odefunc, z0, samp_ts)
            pred_z = pred_z.detach().numpy()
            if args.extra_dim == 0:
                name = 'figure_data./node_film_2d.npy'
            else:
                name = 'figure_data./anode_film_(2+'+str(args.extra_dim)+')d.npy'
            np.save(name, pred_z)
        elif args.data_dimension == 3:
            z0 = torch.tensor(np.load('data./vis_data./3d_vis_data.npy')).float()
            ntotal = len(z0)
            #Augment z0
            zeros = torch.zeros(ntotal, args.extra_dim).float()
            z0 = torch.cat((z0, zeros), dim=1).to(device)
            pred_z = odeint(feature_layers[0].odefunc, z0, samp_ts)
            pred_z = pred_z.detach().numpy()
            if args.extra_dim == 0:
                name = 'figure_data./node_film_3d.npy'
            else:
                pass
            np.save(name, pred_z)
        else:
            pass

    


