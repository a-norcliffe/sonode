import argparse
import numpy as np
import torch
import torch.nn as nn



parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--noscillators', type=int, default=30)
parser.add_argument('--ntimestamps', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--omega', type=float, default=1.0)
args = parser.parse_args()

from torchdiffeq import odeint_adjoint as odeint



class ODEfunc(nn.Module):

    def __init__(self):
        super(ODEfunc, self).__init__()
        self.gamma = args.gamma
        self.omega_sq = (args.omega)**2

    def forward(self, t, z):
        cutoff = int(len(z)/2)
        x = z[:cutoff]
        v = z[cutoff:]
        out = -2*self.gamma*v -(self.gamma**2+self.omega_sq)*x
        return torch.cat((v, out))
    
    

if __name__ == '__main__':
    
    func = ODEfunc()
    
    #make initial conditions
    z0 = (np.random.rand(int(args.noscillators*2))-0.5)*5
    z0 = torch.tensor(z0).float()
    z0 = z0.reshape((int(args.noscillators*2), 1))
    
    # model
    # making time samples
    t0, tN = 0, 10
    samp_ts = torch.linspace(t0, tN, args.ntimestamps)

    z = odeint(func, z0, samp_ts)
    
    np.save('data./z0.npy', z0.numpy())
    np.save('data./z.npy', z.numpy())
    np.save('data./samp_ts.npy', samp_ts.numpy())

    

       
        

        
    

        
      

        
    


