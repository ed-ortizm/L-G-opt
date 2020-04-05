## I'll be following these two tutorials
# http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing
# https://perso.crans.org/besson/publis/notebooks/Simulated_annealing_in_Python.html
# Here some important points
    # If the change in energy is negative, the energy state of the new
    # configuration is lower and the new configuration is accepted.
    # I'll be using the Boltzman factor unless Jeremy tells me something else.
    # If the change in energy is positive, the new configuration has a
    # higher energy state; however, it may still be accepted according
    #to the Boltzmann probability factor
    # The Boltzmann probability is compared to a random number drawn from a
    # uniform distribution between 0 and 1; if the random number is smaller
    # than the Boltzmann probability, the configuration is acceptedself.
    # This allows the algorithm to escape local minima.
import numpy as np
from math import sin, pi, log, exp
import matplotlib
import matplotlib.pyplot as plt
## energy: in my case it will be the original function with a minus
# just to convert the maxima into a minima (cp & pt from lm00 )

def energy(x,nn=1):
    n = nn
    # I add the minus sign to cenvert the problem in a minimization
    # problem, to feel familiar with a sytem looking for the minimum energy.
    f = (16*x[0]*(1-x[0])*x[1]*(1-x[1])*sin(n*pi*x[0])*sin(n*pi*x[1]))**2
    return -f

def acceptance_p(dE,T, dE_avg):
    B_factor = exp(-dE / (dE_avg * T) )
    return B_factor

def annealing(x_start=[0.,0.],n=50,m=50,nn=1):
    print(x_start)
    # number of accepted solutions
    n_acc = 0.
    # Probability of acceptin worst solution at the beginning and in the end
    p_0 = 0.7
    p_n = 0.001
    # Initial and final Temperature
    T_0 = -1./log(p_0)
    T_n = -1./log(p_n)
    # Temperature reduction for each step in the walk
    T_frac = (T_n/T_0)**(1.0/(n-1.0))
    ## Numpy array to store the values for latter plotting results
    x = np.zeros((n+1,2))
    x[0] = x_start
    # Updating value during iterations
    x_i  = [0,0]
    n_aa = n_acc + 1 # I automatically accept the starting point
    # Current optimum point
    x_c = x_start # Again the starting point is initially accepted
    energy_c = energy(x_i,nn) # Again... that's why I use _i
    energies = np.zeros(n+1)
    energies[0] = energy_c # Again... that's why I use _i
    # Current temperature
    T = T_0
    # k_b equivalent
    dE_avg = 0.0
    for i in range(n):
        for j in range(m):
            # Generating new point
            x_i[0] = x_c[0] + np.random.random() - 0.5
            x_i[1] = x_c[1] + np.random.random() - 0.5
            aux = np.array(x_i)
            # Making sure the new point is in the unit sqare with np.clip()
            x_i[0] = np.clip(aux,0.,1.)[0]
            x_i[1] = np.clip(aux,0.,1.)[1]
            # Change in energy
            dE = abs(energy(x_i,nn)-energy_c)
            if  energy(x_i,nn) < energy_c:
                x_c[0], x_c[1] = x_i[0], x_i[1]
                energy_c = energy(x_i,nn)
                n_acc = n_acc + 1.
                dE_avg = (dE_avg*(n_acc-1) + dE)/n_acc
            else:
                if (i==0) and (j==0):
                    dE_avg = dE
                p = acceptance_p(dE,T,dE_avg)
                if np.random.random() < p:
                    x_c[0], x_c[1] = x_i[0], x_i[1]
                    energy_c = energy(x_i,nn)
                    n_acc = n_acc + 1.
                    dE_avg = (dE_avg*(n_acc-1) + dE)/n_acc
        # Best values after a thermalization
        x[i+1][0]     = x_c[0]
        x[i+1][1]     = x_c[1]
        energies[i+1] = energy_c
        # Reducing the temperature
        T = T_frac * T
    return energies,x

def annealing_plot(points, energies, x_s, nn,chain):
    #opt = 0.5*np.ones(energies.size)
    plt.figure()
    plt.suptitle('x and y starting at ' + x_s + \
    'for n = ' + str(nn))
    plt.subplot(121)
    plt.ylim(-0.1,1.1)
    plt.hlines(0.5,0,energies.size)
    plt.plot(points[:,0],'r')
    #plt.plot(points[:,0],opt, 'r')
    plt.title("x")
    plt.subplot(122)
    plt.ylim(-0.1,1.1)
    plt.hlines(0.5,0,energies.size)
    plt.plot(points[:,1],'b')
    #plt.plot(points[:,1],opt, 'b')
    plt.title("y")
    plt.savefig('xy_evolution_n_'+ str(nn) + 'chain_' + str(chain) + '.png')

    plt.figure()
    plt.title("energy for n = " + str(nn) + ' starting at ' + x_s)
    plt.ylim(-0.1,1.1)
    plt.hlines(1.,0,energies.size)
    plt.plot(-1.*energies,'b')
    plt.savefig('energy_n_' + str(nn)+ 'chain_' + str(chain) +'.png')
