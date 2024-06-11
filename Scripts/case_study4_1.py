import numpy as np
import tensorly as tl
import tensorly.contrib.sparse.decomposition as tl_sparse
from scipy.integrate import odeint
from itertools import product
from scipy.optimize import root
from scipy.sparse import coo_array
import matplotlib.pyplot as plt
#import tensorflow as tf
import MTI
import Model_MTI
import tensor_methods as methods
import methods_linearize as l_methods
import numerical as num
import time
import sparse
import itertools
import optimisation as optim
import tensorly.contrib.sparse as tlsp
#np.random.seed(42) 
#We define grid parameters
L = 50
R = 0.5
C = 20

verbose = True
T = 100
#Here x = (i i'), u = (v v')
n_nonbin = 5
n_bin = 2
n = n_bin + n_nonbin

#We define a non linear (neither Multi Linear) time indepedenment EDO system
#We first set up the tensor related to the problem
#And we add auxiliary variable v s.t y5 = v
e = n
order = 2
F_shape = (1+n,)*order + (n_nonbin,)
F = np.zeros(F_shape)
F = np.random.rand(*F_shape) - 0.5*np.ones(F_shape)   
F_lim_indices = (slice(1+n),)*order + (slice(e),)
F_lim = F[F_lim_indices]

v_0 = np.random.rand(1)
def binary_EDO(vector, t):
    res = np.zeros(3)
    v = v_0 
    v_abs = vector[0] 
    v_plus = vector[1] 
    v_minus = vector[2]
     
    res[0] = (v_abs-v)*(v_abs+v)
    res[1] = v_plus*v_minus
    res[2] = v_plus + v_minus - v
    return res

aux = 10*np.random.rand(1)
vector_0 = np.array([np.abs(v_0), max(v_0, 0), v_0])
vector_0 = vector_0.flatten()
sol = root(binary_EDO, vector_0, args=(1))
print(sol)
print(binary_EDO(sol.x, 1))
                
#We now consider the Algebraic part by defining the apropriate tensor G
e = n
e = 0
G_shape = (n+1,)*order + (n_bin,)