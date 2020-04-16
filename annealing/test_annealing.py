#!/usr/bin/env python3
import sys
from annealing import *
## Initial parameters for the annealing
# n: number of steps, also length of the chain
n = int(sys.argv[1])
# m: number of microstates visited to ensure thermalization
m = int(sys.argv[2])
# Number of starting points
p = int(sys.argv[3])
# chains   = np.zeros((n+1,2*p))
# print(chains.shape)
# nn: integer controling the number of picks in my energy
#nn = int(sys.argv[3])
# x: the starting point
#x_start = float(sys.argv[3])
#y_start = float(sys.argv[4])
#x       = np.array([x_start,y_start])

# convergence criteria
e = 0.05 # 0.01 is to much to ask to this algorithm
#### Starting annealing
## Defining different values 'nn' for the energy function
nns = [2*i+1 for i in range(11)]
## Defining 10 random starting points to get 10 chains
runs = 5
conv_ratios = np.zeros((len(nns),runs+1))
for run in range(runs):
    print('run: ', run)
    conv_its = 0
    chains_x = np.zeros((n+1,p))
    chains_y = np.zeros((n+1,p))
    # n+1 is to include the starting point
    i = 0 # idx moving on nns, to store the conv ratios as a function of nn
    for nn in nns:
        xx = np.random.random((p,2))
        # 1 if converged, 0 if not
        convergence = np.zeros(xx.shape[0])
        n_chain = 0
        means= np.zeros(xx.shape)
        variances = np.zeros(xx.shape)
        for x in xx:
            # x_s is the starting point in string format, used for naming files
            # and the title
            x_s = '('+str(x[0])[:6]+','+str(x[1])[:6]+')'
            #print('starting at ', x)
            energies, points  = annealing(x_start=x ,n=n,m=m, nn = nn)
            # chains[:,2*n_chain]   = points[:,0]
            # chains[:,2*n_chain+1] = points[:,1]
            chains_y[:,n_chain]   = points[:,1]
            chains_x[:,n_chain]   = points[:,0]
            #print('ending at', points[-1])
            #print('final energy ', energy(points[-1],nn))
            # Checking for convergence
            if abs(1+energy(points[-1],nn))< e:
                convergence[n_chain] = 1
                #print('Convergence for chain ', n_chain)
            # Saving the mean and variance for each chain
            # idx = n*2//3
            # means[n_chain][0] = points[idx:].mean(axis=0)[0]
            # means[n_chain][1] = points[idx:].mean(axis=0)[1]
            # variances[n_chain][0] = points[idx:].var(axis=0)[0]
            # variances[n_chain][1] = points[idx:].var(axis=0)[1]
            n_chain = n_chain + 1
        np.savetxt('chains_x_n_' + str(nn) + '_run_' + str(run) + '.txt' ,\
        chains_x, delimiter='\t', fmt="%1.4f")
        np.savetxt('chains_y_n_' + str(nn) + '_run_' + str(run) + '.txt' ,\
        chains_y, delimiter='\t', fmt="%1.4f")
        # np.savetxt('chains_n_' + str(nn) , chains_y, delimiter='\t', fmt="%1.4f")
            #annealing_plot(points,energies,x_s=x_s,nn=nn,chain = n_chain)
        ## Computing convergence rate
        # print('convergence rate for nn ', nn)
        conv_rate = convergence.sum()/len(convergence)
        conv_ratios[i][0] = nn
        conv_ratios[i][run+1] = conv_rate
        # print(conv_rate,conv_ratios[i][0],conv_ratios[i][1])
        i = i+1
        ## https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
        # Computing the overall sample posterior mean, the between-chains and the
        # within-chain variances
        # chains = xx.shape[0]
        # mean = means.mean(axis=0)
        # means22 = means**2
        # variances22 = variances**2
        # B = (m/(chains-1))*(means22.sum(axis=0)+chains*mean**2-2*mean*means.sum(axis=0))
        # W = (1/chains)*(variances22.sum(axis=0))
        # # print('for nn ', nn, ' we have')
        # # print(B,W)
        #
        # # Potential Scale Reduction Factor
        #
        # PSRF = ((m-1)/m) + ((chains+1)/(chains*m))*B
        # # print('The PSRF is ', PSRF)
np.savetxt('conv_rates.txt', conv_ratios, delimiter='\t', fmt="%1.2f")
