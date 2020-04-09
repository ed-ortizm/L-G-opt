#!/usr/bin/env python3
import sys
from annealing import *
## Initial parameters for the annealing
# n: number of steps
n       = int(sys.argv[1])
# m: number of microstates visited to ensure thermalization
# equivalent to the length of the chain.
m       = int(sys.argv[2])
# nn: integer controling the number of picks in my energy
#nn = int(sys.argv[3])
# x: the starting point
#x_start = float(sys.argv[3])
#y_start = float(sys.argv[4])
#x       = np.array([x_start,y_start])

#### Starting annealing
## Defining different values 'nn' for the energy function
nns = [2*i+1 for i in range(2)]
## Defining 10 random starting points to get 10 chains
for nn in nns:
    xx = np.random.random((10,2))
    n_chain = 0
    means= np.zeros(xx.shape)
    variances = np.zeros(xx.shape)
    for x in xx:
        # x_s is the starting point in string formar, used for naming files and title
        x_s = '('+str(x[0])[:6]+','+str(x[1])[:6]+')'
        print('starting at ', x)
        energies, points = annealing(x_start=x ,n=n,m=m, nn = nn)
        print('ending at', points[-1])
        # Saving the mean and variance for each chain
        means[n_chain][0] = points.mean(axis=0)[0]
        means[n_chain][1] = points.mean(axis=0)[1]
        variances[n_chain][0] = points.var(axis=0)[0]
        variances[n_chain][1] = points.var(axis=0)[1]
        n_chain = n_chain + 1
        annealing_plot(points,energies,x_s=x_s,nn=nn,chain = n_chain)
    ## https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
    # Computing the overall sample posterior mean, the between-chains and the
    # within-chain variances
    chains = xx.shape[0]
    mean = means.mean(axis=0)
    means22 = means**2
    variances22 = variances**2
    B = (m/(chains-1))*(means22.sum(axis=0)+chains*mean**2-2*mean*means.sum(axis=0))
    W = (1/chains)*(variances22.sum(axis=0))
    # print('for nn ', nn, ' we have')
    # print(B,W)

    # Potential Scale Reduction Factor

    PSRF = ((m-1)/m) + ((chains+1)/(chains*m))*B
    print('The PSRF is ', PSRF)
