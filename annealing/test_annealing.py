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

# convergence criteria
e = 0.05 # 0.01 is to much to ask to this algorithm
#### Starting annealing
## Defining different values 'nn' for the energy function
nns = [2*i+1 for i in range(11)]
# number of times the code will run
runs = 1
# storing the convergence ratios per nn per run
conv_ratios = np.zeros((len(nns),runs+1))
for run in range(runs):
    print('run: ', run)
    # length of the chain when it converges
    conv_lgth = 0
    # Storing the chains
    chains_x = np.zeros((n+1,p))
    chains_y = np.zeros((n+1,p))
    # n+1 is to include the starting point
    i = 0 # idx moving on nns, to store the conv ratios as a function of nn
    for nn in nns:
        xx = np.random.random((p,2))
        # 1 if converged, 0 if not
        convergence = np.zeros(xx.shape[0])
        # idx to flag the number of the chain we are working with
        n_chain = 0
        for x in xx:
            # x_s is the starting point in string format, used for naming files
            # and the title
            x_s = '('+str(x[0])[:4]+','+str(x[1])[:4]+')'
            #print('starting at ', x)
            energies, points  = annealing(x_start=x ,n=n,m=m, nn = nn)
            ## Plotting (only for the first run, could be any run)
            if run==0:
                annealing_plot(points,energies,x_s=x_s,nn=nn,chain = n_chain)
            chains_y[:,n_chain]   = points[:,1]
            chains_x[:,n_chain]   = points[:,0]
            #print('ending at', points[-1])
            #print('final energy ', energy(points[-1],nn))
            # Checking for convergence
            if abs(1+energy(points[-1],nn))< e:
                convergence[n_chain] = 1
                #print('Convergence for chain ', n_chain)
            n_chain = n_chain + 1
        np.savetxt('chains_x_n_' + str(nn) + '_run_' + str(run) + '.txt' ,\
        chains_x, delimiter='\t', fmt="%1.4f")
        np.savetxt('chains_y_n_' + str(nn) + '_run_' + str(run) + '.txt' ,\
        chains_y, delimiter='\t', fmt="%1.4f")
        ## Computing convergence rate
        conv_rate = convergence.sum()/len(convergence)
        conv_ratios[i][0] = nn
        conv_ratios[i][run+1] = conv_rate
        # print(conv_rate,conv_ratios[i][0],conv_ratios[i][1])
        i = i+1
np.savetxt('conv_rates.txt', conv_ratios, delimiter='\t', fmt="%1.2f")
