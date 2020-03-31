import sys
import numpy as np
from math import sin, pi, log, exp

n       = int(sys.argv[1])
x_start = float(sys.argv[2])
y_start = float(sys.argv[3])
x       = np.array([x_start,y_start])
# Number of microstates in the enssemble
m = 50
#m = int(sys.argv[4])
# http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing

# Lets define our function: in my case it will be the original one with a minus
# just to convert the maxima into a minima (cp & pt from lm00 )

def energy(x,n):
    # I add the minus sign to cenvert the problem in a minimization
    # problem, to feel familiar with a sytem looking for the minimum energy.
    f = -(16*x[0]*(1-x[0])*x[1]*(1-x[1])*sin(n*pi*x[0])*sin(n*pi*x[1]))**2
    return f

def acceptance_p(dE,T):

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
    B_factor = exp(-dE/T)
    pass

def annealing(x=x,n=n,m=m):
    # Probability of acceptin worst solution at the beginning and in the end
    p_0 = 0.7
    p_n = 0.001
    # Initial and final Temperature
    T_0 = -1./log(p_0)
    T_n = -1./log(p_n)
    # Fraction reduction for each step
    T_frac = (T_n/T_0)**(1.0/(n-1.0))
    # Temperature
    T = T_0
    x_c = x
    x_i = np.zeros(2)
    dE_avg
    # The first for is for the number of steps my walker will take
    for i in range(n):
        # The second for is to thermalize the ensemble (micro canonico)
        for j in range(m):
            # Generating new point
            x_i[0] = x[0] + np.random.random()
            x_i[1] = x[1] + np.random.random()
            # Making sure the new point is in the unit sqare with np.clip()
            x_i = np.clip(x_i,0.,1.)
            # Change in energy
            dE = energy(x_i) - energy(x_c)
            if dE < 0:
                x_c = x_i
            else:
                if (i==0) and (j==0):
                    dE_avg = dE
                p = acceptance_p(dE,T)
                if np.random.random() < p:
                    x_c = x_i
                else:
                x_c = x_c
            # Reducing the temperature
            T = T_frac * T
    pass

# Starting the code

    # Collecting costs and states to make statistics and plots later on
state = np.array([x_start,y_start])
cost =  f(state,n)
states, costs = [state],[cost]

for step in range(m_steps):
    fraction = step
