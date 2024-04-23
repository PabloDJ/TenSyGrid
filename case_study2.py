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
import methods_linearize as l_methods
import numerical as num
import time
import itertools
#np.random.seed(42) 
#We define grid parameters
L = 50
R = 0.5
C = 20


T = 100
t = range(T)
#Here x = (i i'), u = (v v'

n = 3
A = 5*np.random.rand(n,n,n)
#We define a non linear (neither Multi Linear) time indepedenment EDO system
def f_EDO(y, t):
    dydt = np.zeros(n)
    y = y.reshape(-1, 1) 
    for i in range(n):
        dydt[i] = y.T@A[i]@y
    return dydt

#We first set up the tensor related to the problem
#And we add auxiliary variable v s.t y5 = v
e = n
F_shape = (2,)*n + (2,)*n + (e,)
F = np.zeros(F_shape)
for i in range(n):
    A_i = A[i]
    F_index = [0]*n + [0]*n + [i]
    for j,k in itertools.product(range(0, n), repeat=2):
        F_index[j] = 1 
        F_index[k] = 1 
        if j!= k:
            F[tuple(F_index)] = A_i[j,k]
        else:
            F_index[n+k] = 1
            F[tuple(F_index)] = A_i[j,k]
 
e = n
G_shape = [2]*n + [2]*n + [e]
G = np.zeros(G_shape)
for i in range(n):
    G_index_x = [0]*n + [0]*n
    G_index_v = [0]*n + [0]*n
    G_index_xv = [0]*n + [0]*n

    G_index_x[i] = 1
    G_index_v[n+i] = 1
    G_index_xv[i] = 1
    G_index_xv[n+i] = 1
    
    G[tuple(G_index_x) + (i,)] = 1
    G[tuple(G_index_v) + (i,)] = -1
#We now consider the Algebraic part

F_CP_tensor = tl.decomposition.parafac_power_iteration(tl.tensor(F), rank=15) 
F_CP_tensor = MTI.Tensor_decomposition(F_CP_tensor[1],F_CP_tensor[0])
#F_CP_tensor = tl.decomposition.symmetric_parafac_power_iteration(tl.tensor(F), rank=10) 
#F_CP_tensor = tl.decomposition.non_negative_parafac(tl.tensor(F), rank=10) 
#F_CP_tensor = tl.decomposition.parafac(tl.tensor(F), rank=10) 
#G_CP_tensor = tl.decomposition.parafac(tl.tensor(G), rank=10) 
G_CP_tensor = tl.decomposition.parafac_power_iteration(tl.tensor(G), rank=10) 
G_CP_tensor = MTI.Tensor_decomposition(G_CP_tensor[1],G_CP_tensor[0])

eMTI = MTI.eMTI(n, 0, 0, F, G)
axis = [i for i in range(2*n)]
f1 = lambda x: np.tensordot(F, l_methods.monomial(x, tensorize = True), axes=(axis, axis))
f2 = lambda x: methods.CP_MTI_product(F_CP_tensor, x)
f3 = lambda x: f_EDO(x[:n], 1)

x_0 = np.random.rand(2*n)
u = np.random.rand(110,110)
dt = 0.1
t = np.linspace(0, 0.2, 10)
f1_eval = f1(x_0)
f2_eval = f2(x_0)
f3_eval = f3(x_0)
print(f"f1 eval {f1_eval}")
print(f"f2 eval {f2_eval}")
print(f"f3 eval {f3_eval}")

raise Exception("stop the count")
EDO_start = time.time()
print("sol 1")
sol = odeint(f_EDO, x_0, t)
print("sol 2")
sol_trap = num.trapezoidal_implicit(f_EDO, x_0, t)
print("solution is", sol)
EDO_time = time.time() - EDO_start

#WE have system st x = (xdot1, x1, u1, u2, v1, v2) dxdt = (x)'
def DAE_f(x, z, t):
    xaux = np.concatenate((x,z))
    res = methods.CP_MTI_product(F_CP_tensor, xaux)
    return res

def DAE_g(x, z, t):
    xaux = np.concatenate((x,z))
    res = methods.CP_MTI_product(G_CP_tensor, xaux)
    #We add the constraint  
    return res

DF = methods.diffenrentiation_CP(F_CP_tensor)
DG = methods.diffenrentiation_CP(G_CP_tensor)

def DAE_J(x, z, t):
    xaux = np.concatenate((x,z))
    partialx_F = methods.compute_diff_CPx(DF, xaux)
    partialx_G = methods.compute_diff_CPx(DG, xaux)
    J = np.concatenate((partialx_F, partialx_G))
    return J


z_0 = x_0
DAE_start = time.time()
x_sol, z_sol = num.backward_euler_semi_explicit(DAE_f, DAE_g, x_0, z_0, t)
DAE_time = time.time()- DAE_start
DAE2_start = time.time()
x_solbis, z_solbis = num.backward_euler_semi_explicit(DAE_f, DAE_g, x_0, z_0, t, DAE_Jacob = DAE_J)
DAE2_time = time.time() - DAE2_start

color_codes = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
print(f"time EDO is {EDO_time}")
print(f"time DAE is {DAE_time}")
print(f"time DAE2 is {DAE2_time}")

for i in [0,1]:
    plt.plot(t, sol[:, i], color_codes[0], label= f'y{i} EDO')
    plt.plot(t, x_sol[:, i], linestyle=':', color= color_codes[1], label=f'y{i} DAE')
    plt.plot(t, x_solbis[:, i], linestyle='--', color= color_codes[2], label=f'y{i} DAE + Jacob')
    print("diff", sum(np.abs(sol[:, i]-x_solbis[:,i])))
    print("diff", sum(np.abs(x_sol[:, i]-x_solbis[:,i])))
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.show()


