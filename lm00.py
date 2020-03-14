import numpy as np
from math import sin,cos,pi
x = np.linspace(0.,1.,1000)
y = np.linspace(0.,1.,1000)
#3Dploting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# p --> free parameters of the fuction: it is a list
# dom --> domain of the fuction, it is a numpy array
# data --> it is the data to be fitted. Note: if you are testing, you can generate data with the fuction plus some noise --> data =  f + noise
# n --> it s the number controlling the number of extrema in the fucntion
class Function:
    def __init__(self, n = 1):
        self.n = n
        #self.x = x
        #self.y = y
        # Grid evaluation for ploting
        #self.X,self.Y = np.meshgrid(self.x, self.y)
    def eval(self,x,y):
        f = (16*x*(1-x)*y*(1-y)*sin(self.n*pi*x)*sin(self.n*pi*y))**2
        return f

    def geval(self,x,y):
# because of the np.meshgrid I get z = [f([x1,:]),f([x2,:]),...]
        # Grid evaluation for ploting
        X,Y = np.meshgrid(x, y)
        g_f = (16*X*(1-X)*Y*(1-Y)*np.sin(self.n*np.pi*X)*np.sin(self.n*np.pi*Y))**2
        return g_f
# Since our function is an scalar function, the Jacobian is equivalent to the gradient
    def jacobian(self):
# Because of np.meshgrid I get J = [gx_f([x1,:]),gx_f([x2,:]),..., gy_f([x1,:]),gx_f([x2,:])]
        J = np.gradient(self.geval())
        return np.array(J)

    def plt2(self):
        x = np.linspace(0.,1.,1_000)
        y = np.linspace(0.,1.,1_000)
        X,Y = np.meshgrid(x,y)
        z = self.geval(x,y)
        fig,ax = plt.subplots(1,1)
        cp = ax.contourf(X, Y, z)
        fig.colorbar(cp)
        plt.show()

    def plt3(self):
        x = np.linspace(0.,1.,1_000)
        y = np.linspace(0.,1.,1_000)
        X,Y = np.meshgrid(x,y)
        z = self.geval(x,y)

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
    J = z.jacobian()
    J_t = np.transpose(J)
    JtJ = np.matmul(Jt,J)

    # Damping factor
    mu = tau * np.diagonal(JtJ).max() # intuitively since JtJ is related to the hessian, it takes into account
    # the curvature (??)

# The maximun value of my function approximately 1 --> self.eval().max()=0.9995955408159843
    e = z - np.ones(z.shape)
    e = epsilon**2 # numpy sqaures element-wise!
    # initial movement
    delta = np.random.random(2)
    # Augmented normal equation N = diag(mu) + J(1-z)
    N = np.zeros(JtJ.shape)
    np.fill_diagonal(N,1.)
    N = mu * N
    pass
    return xy_opt
