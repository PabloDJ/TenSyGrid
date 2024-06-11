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
import scipy.io
#np.random.seed(42) 
#We define grid parameters
L = 50
R = 0.5
C = 20

verbose = True
T = 100
#Here x = (i i'), u = (v v')
sorted(i for i in set(dir(tl)).intersection(dir(tlsp))
       if not i.startswith('_'))
print(tl.get_backend())
n = 80


#We define a non linear (neither Multi Linear) time indepedenment EDO system
#We first set up the tensor related to the problem
#And we add auxiliary variable v s.t y5 = v
e = n
order = 2
F_shape = (1+n+n,)*order + (e,)
F_shape = (1+n,)*order + (e,)
F = np.zeros(F_shape)
F = np.random.rand(*F_shape) - 0.5*np.ones(F_shape)   
F_lim_indices = (slice(1+n),)*order + (slice(e),)
F_lim = F[F_lim_indices]

def f_EDO(y, t):
    dydt = np.zeros(n)
    dydt = methods.MULT_polynomialtensor(F_lim, y)
    return dydt
                
#We now consider the Algebraic part by defining the apropriate tensor G
e = n
e = 0
G_shape = (1+n+n,)*order + (e,)
G = np.zeros(G_shape)

#Not necessary given that F takes the quadric parameters into account.
for i in range(n):
    break
    G[i, 0, i] = 1
    G[0, i, i] = 1
    G[n+i, 0, i] = -1
    G[0, n+i, i] = -1
G = np.random.rand(*G_shape)    

#WE CHOSE ONE OF THE FOLLOWING DECOMPOSITION METHODS
F_CP_tensor = tl.decomposition.parafac(tl.tensor(F), rank= 100) 
try:
    G_CP_tensor = tl.decomposition.parafac(tl.tensor(G), rank=100) 
except:
    a = 0
#IF NOT PARAFAC UNCOMMENT THE FOLLOWING 2 LINES
#F_CP_tensor = MTI.Tensor_decomposition(F_CP_tensor[1],F_CP_tensor[0])
#G_CP_tensor = MTI.Tensor_decomposition(G_CP_tensor[1],G_CP_tensor[0])

eMTI = MTI.eMTI(n, 0, 0, F, G)
axis = [i for i in range(2*n)]
f1 = lambda x: methods.MULT_polynomialtensor(F, x)
f2 = lambda x: methods.CP_MULT_polynomialtensor(F_CP_tensor, x)
f3 = lambda x: f_EDO(x[:n], 1)

print("axis is ", axis)
x_0 = np.random.rand(n)
z_0 = np.array([])
xz_0 = np.concatenate((x_0, z_0))
print("shape xz ", xz_0.shape)
print("shape F", F.shape)
u = np.random.rand(110,110)
dt = 0.1
t = np.linspace(0, 0.2, 100)
f1_eval = f1(xz_0)
f2_eval = f2(xz_0)
f3_eval = f3(xz_0)
print(f"f1 eval {f1_eval}")
print(f"f2 eval {f2_eval}")
print(f"f3 eval {f3_eval}")

EDO_start = time.time()
sol = odeint(f_EDO, x_0, t)
EDO_time = time.time() - EDO_start
sol_trap = num.trapezoidal_implicit(f_EDO, x_0, t)

#WE have system st x = (xdot1, x1, u1, u2, v1, v2) dxdt = (x)'
def DAE_f(x, z, t):
    xaux = np.concatenate((x,z))
    res = methods.CP_MULT_polynomialtensor(F_CP_tensor, xaux)
    return res

def DAE_g(x, z, t):
    xaux = np.concatenate((x,z))
    try:
        res = methods.CP_MULT_polynomialtensor(G_CP_tensor, xaux)
    except:
        res = np.array([])
    #We add the constraint  
    return res

#DF = methods.diffenrentiation_CP(F_CP_tensor)
#DG = methods.diffenrentiation_CP(G_CP_tensor)

def DAE_J(x, z, t):
    xaux = np.concatenate((x,z))
    partialx_F = methods.compute_diff_CPx(DF, xaux)
    partialx_G = methods.compute_diff_CPx(DG, xaux)
    J = np.concatenate((partialx_F, partialx_G))
    return J

DAE_start = time.time()
#x_sol, z_sol = num.backward_euler_semi_explicit(DAE_f, DAE_g, x_0, z_0, t)
DAE_time = time.time()- DAE_start
DAE2_start = time.time()
x_solbis, z_solbis = num.backward_euler_semi_explicit(DAE_f, DAE_g, x_0, z_0, t, DAE_Jacob = False)
DAE2_time = time.time()- DAE2_start
DAE3_start = time.time()
x_solbis2, z_solbis2 = num.algebraic_substitution_semi_explicit(DAE_f, DAE_g, x_0, z_0, t)
DAE3_time = time.time()- DAE3_start
x_sol, z_sol = (x_solbis, z_solbis )
#x_solbis2, z_solbis2 = (x_solbis, z_solbis)

color_codes = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
print(f"time EDO is {EDO_time}")
print(f"time DAE is {DAE_time}")
print(f"time DAE 2 is {DAE2_time}")
print(f"time DAE algebraic is {DAE3_time}")

for i in [0,1]:
    plt.plot(t, sol[:, i], color_codes[0], label= f'y{i} EDO')
    plt.plot(t, x_sol[:, i], linestyle=':', color= color_codes[1], label=f'y{i} DAE')
    plt.plot(t, sol[:, i] - x_solbis[:, i], linestyle='--', color= color_codes[2], label=f'Difference i')
    print("diff EDO DAE", sum(np.abs(sol[:, i]-x_solbis[:,i])))
    print("diff DAE DAE", sum(np.abs(x_sol[:, i]-x_solbis[:,i])))
    print("diff DAE ALG", sum(np.abs(x_sol[:, i]-x_solbis2[:,i])))
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.show()

verbose = False
if verbose==True:
    for ti in range(len(t)):
        print("timestep is ", ti)
        print(f"x EDO {sol[ti,:]}")
        print(f"dxdt EDO {f_EDO(sol[ti,:],t)}")
        print(f"x DAE {x_sol[ti,:]}")
        print(f"f EDO DAE {f_EDO(x_sol[ti,:],t)}")
        print(f"f1 DAE {f1(np.concatenate((x_sol[ti,:], x_sol[ti,:])))}")
