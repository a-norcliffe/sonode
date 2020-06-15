import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

test_bool = True
npoints = 200
end_time = 200
mu = 8.53
A = 1.4
omega = np.pi/5
z0 = torch.tensor([.1, 0]).float()

if test_bool:
    filename = 'data./test_'
else:
    filename = 'data./'

from torchdiffeq import odeint


def vdp(t, z):
    x = z[0]
    v = z[1]
    out = mu*(1-x**2)*v - x + A*torch.cos(omega*t)
    return torch.tensor([v, out])


times = torch.linspace(0, end_time, npoints)
torch.save(times, filename+'time_data.pt')
times_arr = times.numpy().reshape(npoints)

z = odeint(vdp, z0, times)

idsx = torch.full((npoints, 1), 0).long()
idsv = torch.full((npoints, 1), 1).long()

x = z.gather(1, idsx)
v = z.gather(1, idsv)
torch.save(x, filename+'position_data.pt')
torch.save(v, filename+'velocity_data.pt')


x_arr = x.numpy()
v_arr = v.numpy()

x_arr = x_arr.reshape((npoints))
v_arr = v_arr.reshape((npoints))

plt.figure()
plt.plot(x_arr, v_arr)
plt.figure()
plt.plot(times_arr, x_arr)
plt.scatter(times_arr, x_arr)
plt.figure()
plt.plot(times_arr, v_arr)
plt.scatter(times_arr, v_arr)
