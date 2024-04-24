import numpy as np
import tensorly as tl
from scipy.integrate import odeint
from itertools import product
from scipy.optimize import root
import matplotlib.pyplot as plt
#import tensorflow as tf
import MTI
import Model_MTI
import tensor_methods as methods
import numerical as num
import time

class DAE:
    #We build the DAE class representing a Differential Algebraic equation system
    ### atributes:
    # n integer: length of x , x being the state of the DAE
    # z_dim integer: length of z, z being an auxiliary variable to the DAE
    # e integer: number of algebraic equations in explicit form
    # F numpy array or tensorly tensor: describes the Differential equations of the DAE
    # F dimensions need to match n and z_dim 
    # G numpy array or tensorly tensor: describes the Algebraic equations of the DAE 
    # G dimensions need to match n, z_dim and e
    # H tensor, describe all the DAE's equation in implicit form
    def __init__(self, n, z_dim, e, F, G):  
        self.n = n
        self.z_dim = z_dim
        self.e = e
        self.F = F
        self.G = G

    def build_H(self):
        H_shape = (2,)*self.n + (2,)*self.n + (2,)*self.z_dim + (self.e+self.n,)
        H = tl.zeros(H_shape)
        F_old =  (0,)*self.n + (slice(0,2),)*self.n + (slice(0,2),)*self.z_dim + (slice(0,self.n),)
        G_old =  (0,)*self.n + (slice(0,2),)*self.n + (slice(0,2),)*self.z_dim + (slice(self.n, self.n + self.e),)
        H[F_old] = self.F
        H[G_old] = self.G
        for i in range(i):
            xdot_index = (0,)*(self.n-i) + (1,) + (0,)*(i-1) + (0,)*self.n + (0,)*self.z_dim + (i,)
            H[xdot_index] = -1
        self.H = H
    
    def H_multiply(self):
        return res
    
    def H_CPmultiply(self):
        return res 
    
    def H_diff(self):