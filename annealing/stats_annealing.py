#!/usr/bin/env python3
import numpy as np
def mean_var(data):
    # Saving the mean and variance for each chain
    # Using only the last third of data: the idea is to use only data that is 'burned'
    idx = data.shape[0]*2//3
    data = data[idx:]
    means = np.mean(data, axis=0)
    variances = np.var(data, axis=0)
    return means, variances

def GelmanRubin(means, variances,length):
    N = length
    M = means.shape[0]
    ovl_mean = np.mean(means)
    # Between-chains variances
    B = (means - ovl_mean)**2
    B = (N/(M-1))*B.sum()
    # Between-chains variances
    W = (1/M)*variances.sum()
    # Pooled variance
    V = ((N-1)/N)*W + ((M+1)/(M*N))*B
    # Potential Scale Reduction Factor
    PSRF = V/W
    return PSRF,B, W
# Stats for convergence ratios:

data = np.loadtxt('conv_rates.txt')

for n in data:
    print('For n = ', int(n[0]), ', the convergence ratio has')
    print('A mean value of: ','{:.2f}'.format(np.mean(n[1:])))
    print('A median of : ','{:.2f}'.format(np.median(n[1:])))
    print('A standard deviation of: ','{:.2f}'.format(np.std(n[1:])),'\n')

# GR statistics
# For n =  1 , the convergence ratio has # A mean value of: # 0.88
# For n =  19 , the convergence ratio has # A mean value of: # 0.12

nn = [2*i+1 for i in range(11)]

for n in nn:
    x_file = 'chains_x_n_' + str(n) + '_run_10.txt'
    y_file = 'chains_y_n_' + str(n) + '_run_10.txt'
    x_data = np.loadtxt(x_file)
    y_data = np.loadtxt(y_file)
    x_means, x_variances = mean_var(x_data)
    y_means, y_variances = mean_var(y_data)
    x_PSRF,a,b = GelmanRubin(x_means,x_variances,data.shape[0])
    y_PSRF,a,b = GelmanRubin(y_means,y_variances,data.shape[0])
    print('For n=', n, ', the PSRFs (x,y) are: ', \
    '{:.4f}'.format(x_PSRF),'{:.4f}'.format(y_PSRF))
