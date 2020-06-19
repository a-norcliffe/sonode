Folder containing the files for the [silverbox dataset](http://www.it.uu.se/research/publications/reports/2013-006/2013-006-nc.pdf).

Disclaimer: This experiment is highly dependent on initialisation. Because the ODE is dependent on 
position and velocity, if either of those coefficients is positive, there will be exponential 
growth. Over a "time" range of 1000 this can lead to very large losses and slow training. The
experiment may need to be repeated so that both coefficients initialise as negative. This can
also lead to spikes in the loss during training. Further to this, there is an x^3 term in the ODE. 
When the power is greater than 1, if the coefficient is positive, the solution can diverge to 
infinity in finite "time". This will correspond to underflow in dt. Again this may need the 
experiment to be run multiple times to find an initialisation with negative coefficients. 