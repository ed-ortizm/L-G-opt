import sys
import numpy as np
from math import sin,pi

n = int(sys.argv[1])
x_start = float(sys.argv[2])
y_start = float(sys.argv[3])
m_steps = int(sys.argv[4])
# http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing

# Lets define our function: in my case it will be the original one with a minus
# just to convert the maxima into a minima (cp & pt from lm00 )

def energy(x,n):
        f = -(16*x[0]*(1-x[0])*x[1]*(1-x[1])*sin(n*pi*x[0])*sin(n*pi*x[1]))**2
        return f

def acceptance_p(energy,new_energy,T):
    pass


# Starting the code

    # Collecting costs and states to make statistics and plots later on
state = np.array([x_start,y_start])
cost =  f(state,n)
states, costs = [state],[cost]

for step in range(m_steps):
    fraction = step
