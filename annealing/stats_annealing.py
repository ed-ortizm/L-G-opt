#!/usr/bin/env python3
import numpy as np

# Stats for convergence ratios:

data = np.loadtxt('conv_rates.txt')

for n in data:
    print('For n = ', n[0], ', the convergence ratio has')
    print('A mean value of:')
    print('{:.2f}'.format(np.mean(n[1:])))
    print('A median of :')
    print('{:.2f}'.format(np.median(n[1:])))
    print('A standard deviation of:')
    print('{:.2f}'.format(np.std(n[1:])),'\n')

# GR statistics  
