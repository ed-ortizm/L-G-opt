import numpy as np

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
    def __init__(self, x, y, n = 1):
        self.x = x
        self.y = y
        self.X,self.Y = np.meshgrid(self.x, self.y)
        self.n = n
    def eval(self):
        f = (16*self.X*(1-self.X)*self.Y*(1-self.Y)*np.sin(self.n*np.pi*self.X)*np.sin(self.n*np.pi*self.Y))**2
        return f
# Since our function is an scalar function, the Jacobian is equivalent to the gradient
    def jacobian(self):
        J = np.gradient(self.f)
        return J

    def plt2(self):
        z = self.eval()
        fig,ax = plt.subplots(1,1)
        cp = ax.contourf(self.X, self.Y, z)
        fig.colorbar(cp)
        plt.show()        

    def plt3(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
# Data
        z = self.eval()
# Plot the surface.
        surf = ax.plot_surface(self.X, self.Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
#Plot
        plt.plot()
    
    







