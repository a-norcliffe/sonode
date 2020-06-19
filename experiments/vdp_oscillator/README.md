Folder containing the files for the Van-Der-Pol chaotic oscillator. We have included a 'results.npy'
numpy file, which the programs write the train and test MSE to. However, this was not used in our
experiments.

Disclaimer: This experiment is highly dependent on initialisation. Because the ODE is dependent on 
position and velocity, if either of those coefficients is positive, there will be exponential 
growth. Over a "time" range of 70 this can lead to very large losses and slow training. The
experiment may need to be repeated so that both coefficients initialise as negative. This can
also lead to spikes in the loss during training. In addition, underflow in dt has been observed. 
Therefore, the experiments may need to be repeated.