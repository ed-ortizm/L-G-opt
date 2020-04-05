#!/usr/bin/env python3
import sys
from annealing import *
## Initial parameters for the annealing
# n: number of steps
n       = int(sys.argv[1])
# m: number of microstates visited to ensure thermalization
m       = int(sys.argv[2])
# nn: integer controling the number of picks in my energy
nn = int(sys.argv[3])
# x: the starting point
#x_start = float(sys.argv[3])
#y_start = float(sys.argv[4])
#x       = np.array([x_start,y_start])

#### Starting annealing

## Defining 9 random starting points to get 10 chains
xx = np.random.random((3,2))
n_chain = 0
for x in xx:
    x_s = '('+str(x[0])[:6]+','+str(x[1])[:6]+')'
    n_chain = n_chain + 1
    energies, points = annealing(x_start=x ,n=n,m=m, nn = nn)
    print(x)# cool, here it is returning the values modified by annealing()
    annealing_plot(points,energies,x_s=x_s,nn=nn,chain = n_chain)
