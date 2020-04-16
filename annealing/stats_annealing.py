#!/usr/bin/env python3
import numpy as np
from annealing import GR

# Stats for convergence ratios:
data = np.loadtxt('20_chains_100_runs/conv_rates.txt')

for n in data:
    print('For n = ', int(n[0]), ', the convergence ratio has')
    print('A mean value of: ','{:.2f}'.format(np.mean(n[1:])))
    print('A median of : ','{:.2f}'.format(np.median(n[1:])))
    print('A standard deviation of: ','{:.2f}'.format(np.std(n[1:])),'\n')

# GR statistics
# For n =  1 , the convergence ratio has # A mean value of: # 0.88
# For n =  19 , the convergence ratio has # A mean value of: # 0.12

nn = [2*i+1 for i in range(11)]
runs = 100
x_PSRFs = np.zeros((len(nn),runs+1))
y_PSRFs = np.zeros((len(nn),runs+1))
for run in range(runs):
    i = 0
    for n in nn:
        x_file = '20_chains_100_runs/chains_x_n_' + str(n) + '_run_' + str(run) + '.txt'
        y_file = '20_chains_100_runs/chains_y_n_' + str(n) + '_run_' + str(run) + '.txt'
        x_data = np.loadtxt(x_file)
        y_data = np.loadtxt(y_file)
        # x_means, x_variances = mean_var(x_data)
        # y_means, y_variances = mean_var(y_data)
        # x_PSRF,a,b = GelmanRubin(x_means,x_variances,data.shape[0])
        # y_PSRF,a,b = GelmanRubin(y_means,y_variances,data.shape[0])
        x,y = GR(x_data), GR(y_data)
        x_PSRF,a,b = x.GelmanRubin()
        y_PSRF,a,b = y.GelmanRubin()
        x_PSRFs[i][0] = n
        x_PSRFs[i][run+1] = x_PSRF
        y_PSRFs[i][0] = n
        y_PSRFs[i][run+1] = y_PSRF
        i = i +1
        # print('For n=', n, ', the PSRFs (x,y) are: ', \
        # '{:.4f}'.format(x_PSRF),'{:.4f}'.format(y_PSRF))
np.savetxt('20_chains_100_runs/x_PSRFs.txt',x_PSRFs, delimiter='\t', fmt='%1.4f')
np.savetxt('20_chains_100_runs/y_PSRFs.txt',y_PSRFs, delimiter='\t', fmt='%1.4f')

mx = np.mean(x_PSRFs[:,1:], axis=1)
my = np.mean(y_PSRFs[:,1:], axis=1)

for i in range(len(nn)):
    print('For n=', 2*i+1, ', the mean values of the PSRFs (x,y) are: ', \
    '{:.4f}'.format(mx[i]),'{:.4f}'.format(my[i]))
