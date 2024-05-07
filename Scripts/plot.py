import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vicinity(f, x0, verbose = False):
    n = len(x0)
    vicinity_size = np.ones(n)
    num_points = 100

    # Generate random points in the vicinity of x0
    np.random.seed(0)
    offsets = np.random.uniform( -vicinity_size/2, vicinity_size/2, size=(num_points, n))
    offset_list = []
    for i in range(num_points):
        offset_list.append( offsets[i,:])
    
    x_values = np.ones(offsets.shape) + offsets
    y_values = []
    for i in range(num_points):
        y_values.append(f(x0 + offset_list[i]))
    
    y_values = np.array(y_values)
    
    for i in range(y_values.shape[1]):
        """fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_values[:, 0], x_values[:, 1], c=y_values[:,i], cmap='viridis')
        ax.scatter(x0[0], x0[1], f(x0), color='red', label='$x_0$')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('Function Value')
        ax.set_title('Function Values in the Vicinity of $x_0$')
        plt.legend()
        plt.show()"""
        if verbose:
            plt.figure(figsize=(8, 6))
            plt.scatter(x_values[:, 0], x_values[:, 1], c=y_values[:,i], cmap='viridis')
            plt.colorbar(label='Function Value')
            plt.scatter(x0[0], x0[1], color='red', label='$x_0$')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title('Function Values in the Vicinity of $x_0$')
            plt.legend()
            plt.grid(True)
            plt.show()        
    return