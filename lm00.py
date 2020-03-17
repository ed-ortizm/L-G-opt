import numpy as np
from math import sin,cos,pi,sqrt
from scipy.optimize import approx_fprime as p_grad # point gradient of a scalar funtion
#3Dploting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# p --> free parameters of the fuction: it is a list
# dom --> domain of the fuction, it is a numpy array
# data --> it is the data to be fitted. Note: if you are testing, you can generate data with the fuction plus some noise --> data =  f + noise
# n --> it s the number controlling the number of extrema in the fucntion

# just to avoid repetitive stuff
x = np.linspace(0.,1.,1_000)
X = np.array([x,x])
class Function:
    def __init__(self, n = 1):
        self.n = n
        #self.x = x
        #self.y = y
        # Grid evaluation for ploting
        #self.X,self.Y = np.meshgrid(self.x, self.y)
    def eval(self,x):
        f = (16*x[0]*(1-x[0])*x[1]*(1-x[1])*sin(self.n*pi*x[0])*sin(self.n*pi*x[1]))**2
        return f

    def geval(self,x):
# because of the np.meshgrid I get z = [f([x1,:]),f([x2,:]),...]
        # Grid evaluation for ploting
        X,Y = np.meshgrid(x[0], x[1])
        g_f = (16*X*(1-X)*Y*(1-Y)*np.sin(self.n*np.pi*X)*np.sin(self.n*np.pi*Y))**2
        return g_f
# Since our function is an scalar function, the Jacobian is equivalent to the gradient
#    def jacobian(self):#only data case
# Because of np.meshgrid I get J = [gx_f([x1,:]),gx_f([x2,:]),..., gy_f([x1,:]),gx_f([x2,:])]
#        J = np.gradient(self.geval())
#        return np.array(J)

    def jacobian(self,xy):#only data case
# My function is an scalar, then the JAcobian reduces to the gradient, and here I need a 2D vector
        e = 1e-8*np.ones(2)
        J = p_grad(xy,self.eval,e)
        return J


    def plt2(self):
        x = np.linspace(0.,1.,1_000)
        XY = np.array([x,x]) # new rules, now eval gets one array
        X,Y = np.meshgrid(XY[0],XY[1])
        z = self.geval(XY)
        fig,ax = plt.subplots(1,1)
        cp = ax.contourf(X, Y, z)
        fig.colorbar(cp)
        plt.show()

    def plt3(self):
        x = np.linspace(0.,1.,1_000)
        xy = np.array([x,x]) # new rules, now geval gets one array
        X,Y = np.meshgrid(xy[0],xy[1])
        z = self.geval(xy)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
# Plot the surface.
        surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
#Plot
        plt.plot()

# Defining LM algorithm

def lm(p0,f,k=100):
# P0 is the starting point
p = p0
# e stands for epsilon
# Stop variables
    # k is the maximun number of iterations allowed
    # e1 --> threshold for the gradient
    e1 = 1e-15
    # e2 --> threshold for the change in magnitude of the step |p_f -p_0|
    e2 = 1e-15
    # e3 --> threshold for the sqared error e**2
    e3 = 1e-15
# Updating variables
    # nu --> pase for updating the damping constant mu
    nu = 2.
    # tau --> scaling factor for mu
    tau = 1e-3
# Setting the damping factor
    # mu is the damping factor --> mu = tau*max(JtJ_ii)

    # Jacobian
    J = f.jacobian(p0) # J is a one dimensional array in my case
    JtJ = np.outer(J,J) # np.outer(a,b) trasposes a (colum) so I get a matrix as a result

    # Damping factor
    mu = tau * np.diagonal(JtJ).max() # intuitively since JtJ is related to the hessian, it takes into account
    # the curvature (??)

# The maximun value of my function approximately 1 --> self.eval().max()=0.9995955408159843
    #e = z - np.ones(z.shape)
    #e = epsilon**2 # numpy sqaures element-wise!
    e = 1 - z
    ee = e**2
    # initial movement: random floats in the half-open interval [0.0, 1.0). # 2D in my case
    delta = np.zeros(2)
    # Augmented normal equation N = diag(mu) + JtJ
    N = np.zeros(JtJ.shape)
    np.fill_diagonal(N,1.)
    N = mu * N + JtJ
    # Testing initial values of the displacement
    stop = ee*np.sum(Jt*Jt)
    if  stop < e1:
        print("problem solved")
        break
    k_i = 0
# emulating a do while
    while !stop & k_i < k:
        k_i = k + 1
        while True:
            delta = np.linalg.solve(N,e*Jt)
            delta_module = np.sum(delta*delta)
            p_module = np.sum(p*p)
            if delta_module <= p_module :
                stop = True
            else:
                p_new = p + delta

check @ for matrix mult
        if :
# end of emulating a do while
    pass
    return xy_opt
