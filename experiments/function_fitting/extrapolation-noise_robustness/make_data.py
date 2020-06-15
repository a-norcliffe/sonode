#make data
#manually need to change noise and experiment_no

import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--evenly_spread_timestamps', type=eval, default=True)
parser.add_argument('--noise', type=float, default=0.6) #0.0 gives no noise
parser.add_argument('--npoints', type=int, default=50)
parser.add_argument('--experiment_no', type=int, default=3)
args = parser.parse_args()

        
    
def function(t):
    return torch.sin(t)
    


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_dim = 1
    dim = data_dim
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
    if args.noise != 0.0:
        noise = z.data.new(z.size()).normal_(mean=0, std=args.noise)
        measured_z = (z + noise).to(device)
    else:
        measured_z = z.to(device)   

    torch.save(measured_z, 'data./noisy_sine_data_experiment_'+str(args.experiment_no)\
               +'_noise_'+str(args.noise)+'.pt')
