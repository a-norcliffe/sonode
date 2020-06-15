Programs to run the nested-n-spheres problem, originally called the g mapping from the ANODE paper. The name nested-n-spheres is taken
from Dissecting Neural ODEs. The problem consists of two regions, a red region surrounding a blue region. The red region entirely
surrounds the blue region in n dimensions. The blue and red regions are mapped differently to -1 and +1 respectively, requiring linear
separability. However, NODEs preserve the topology of the input space so the blue region cannot escape the red region. We see that
ANODEs and SONODEs can separate the regions. 

The final learnt flow of the points can be saved by setting `args.visualise` to `True`. 