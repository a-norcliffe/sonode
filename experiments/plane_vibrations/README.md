Folder containing the files for the [airplane vibrations dataset](http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/F16/F16Benchmark.pdf). 

The experiment has been run on the first dataset collected for acceleration 2, the end of the plane wing.

Disclaimer: This experiment is highly dependent on initialisation. Because the ODE is dependent on 
position and velocity, if either of those coefficients is positive, there will be exponential 
growth. Over a "time" range of 1000 this can lead to very large losses and slow training. The
experiment may need to be repeated so that both coefficients initialise as negative. This can
also lead to spikes in the loss during training. In addition, the error message has been observed: 
ValueError: cannot convert float NaN to integer. Where t is NaN. Therefore, the experiments may
need to be repeated.