#!/usr/bin/env python3
import numpy as np
from annealing import GR
import matplotlib
import matplotlib.pyplot as plt

# Stats for convergence ratios:
# data = np.loadtxt('20_chains_100_runs/conv_rates.txt')
# cr_mean = np.zeros((data.shape[0],2))
# i = 0
# for n in data:
#     print('For n = ', int(n[0]), ', the convergence ratio has')
#     cr_mean[i][0] = int(n[0])
#     cr_mean[i][1] = np.mean(n[1:])
#     print('A mean value of: ','{:.2f}'.format(cr_mean[i][0]))
#     print('A median of : ','{:.2f}'.format(np.median(n[1:])))
#     print('A standard deviation of: ','{:.2f}'.format(np.std(n[1:])),'\n')
#     i = i+1
# print(cr_mean[:,0])
# plt.figure()
# plt.title('Convergence Ratio')
# plt.ylabel('Convergence Ratio')
# plt.xlabel('n')
# plt.axis([0,22,0,1.0])
# plt.plot(cr_mean[:,0],cr_mean[:,1],'bo--')
# plt.xticks(np.array([0,1,3,5,7,9,11,13,15,17,19,21]))
# plt.savefig('conv_ratio.png')
# plt.close()

# GR statistics
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
plt.figure()
plt.title('Average PSRFs')
plt.ylabel('PSRFs')
plt.xlabel('n')
plt.axis([0,22,1.,5.])
plt.plot(np.arange(1,23,2),mx,'bo--',label= 'x')
plt.plot(np.arange(1,23,2),my,'ro--',label='y')
plt.legend()
plt.xticks(np.arange(1,23,2))
plt.savefig('PSRFs.png')
plt.close()
