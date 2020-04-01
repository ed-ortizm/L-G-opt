import sys
import numpy as np
from math import sin, pi, log, exp

n       = int(sys.argv[1])
m       = int(sys.argv[2])
x_start = float(sys.argv[3])
y_start = float(sys.argv[4])
x       = np.array([x_start,y_start])
# Thermalization variable, there are fluctuations among the average value of the
# energy
#m
#m = int(sys.argv[4])
# http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing

# Lets define our function: in my case it will be the original one with a minus
# just to convert the maxima into a minima (cp & pt from lm00 )

def energy(x,n=1):
    # I add the minus sign to cenvert the problem in a minimization
    # problem, to feel familiar with a sytem looking for the minimum energy.
    f = -(16*x[0]*(1-x[0])*x[1]*(1-x[1])*sin(n*pi*x[0])*sin(n*pi*x[1]))**2
    return f

def acceptance_p(dE,T, dE_avg):

    # http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing

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
    B_factor = exp(-dE / (dE_avg * T) )
    # What about the sum of all probabilities equals one?
    # We call this function when dE < 0, that's why I use abs(dE)
    return B_factor

def annealing(x_start=[0.,0.],n=50,m=50):
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
    x_i  = x_start
    n_aa = n_acc + 1 # I automatically accept the starting point
    # Current optimum point
    x_c = x[0] # Again the starting point is initially accepted
    energy_c = energy(x_i) # Again... that's why I use _i
    energies = np.zeros(n+1)
    energies[0] = energy_c # Again... that's why I use _i
    # Current temperature
    T = T_0
    # k_b equivalent
    dE_avg = 0.0
    for i in range(n):
        print("Starting Thermalization >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ", i)
        # The second for is to thermalize the ensemble (micro canonico)
        for j in range(m):
            print("Thermal >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", j)
            print("Current point:  ", x_c)
            # Generating new point
            x_i[0] = x_c[0] + np.random.random() - 0.5
            x_i[1] = x_c[1] + np.random.random() - 0.5
            aux = np.array(x_i)
            print("One generated point ", x_i)
            # Making sure the new point is in the unit sqare with np.clip()
            x_i[0] = np.clip(aux,0.,1.)[0]
            x_i[1] = np.clip(aux,0.,1.)[1]
            print("Same point after clip ", x_i)
            # Change in energy
            dE = abs(energy(x_i)-energy_c)
            if  energy(x_i) < energy_c:
                print("lower energy")
                x_c[0], x_c[1] = x_i[0], x_i[1]
                energy_c = energy(x_i)
                n_acc = n_acc + 1.
                dE_avg = (dE_avg*(n_acc-1) + dE)/n_acc
            else:
                if (i==0) and (j==0):
                    dE_avg = dE
                p = acceptance_p(dE,T,dE_avg)
                print("Boltzmann factor: ", p)
                if np.random.random() < p:
                    print("\t\tAccepted")
                    x_c[0], x_c[1] = x_i[0], x_i[1]
                    energy_c = energy(x_i)
                    n_acc = n_acc + 1.
                    dE_avg = (dE_avg*(n_acc-1) + dE)/n_acc
                else:
                    print("\t\tNot accepted")
        # Best values after a thermalization
        print("Current accepted point:>>>>>>>>>>>>>>>", x_c)
        x[i+1][0]     = x_c[0]
        x[i+1][1]     = x_c[1]
        energies[i+1] = energy_c
        # Reducing the temperature
        T = T_frac * T
    return energies,x

# Starting annealing

energies, x = annealing(x_start=x ,n=n,m=m)
print("Starting point: ", x[0])
print("Final point: ", x[-1])
#print(points)
