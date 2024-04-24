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

#We define grid parameters
L = 50
R = 0.5
C = 20

#Dimensions Input
#x is the state of size n
#u the input of size m
#y is the output of size p 
n = 2
m = 2
e = 2 
p = 0
T = 100
t = range(T)
#Here x = (i i'), u = (v v')

#we input the equation as tensors
#G of size R^(p x 2^(n+m))
#F of size R^(p x 2^(n+m))
G_size = (2 for _ in range(n+m)) 
G_size = tuple(G_size) + (p,)
F_size = (2 for _ in range(n+m)) 
F_size = tuple(F_size) + (n,)
G = np.zeros(G_size)
F = np.zeros(F_size)

n = 2
#We define a non linear (neither Multi Linear) time indepedenment EDO system
def f_EDO(y, t):
    y1, y2, y3, y4, y5 = y
    dy1_dt = -0.5 * y1 + 0.1 * y2
    dy2_dt = 0.5 * y1 - 0.2 * y2 + 0.1 * y3
    dy3_dt = -0.3 * y3 + 0.05 * y1 * y2
    dy4_dt = 0.4 * y4 - 0.1 * y5**2
    dy5_dt = -0.2 * y5 + 0.1 * y1 * y4
    return [dy1_dt, dy2_dt, dy3_dt, dy4_dt, dy5_dt]

x_0 = np.random.rand(5)
t = np.linspace(0, 10, 100)
EDO_start = time.time()
sol = odeint(f_EDO, x_0, t)
EDO_time = time.time() - EDO_start


#We first set up the tensor related to the problem
#And we add auxiliary variable v s.t y5 = v
n = 6
e = 5
F_shape = (2,)*n + (e,)
F = np.zeros(F_shape)
F[0,1,0,0,0,0,0] = 0.1
F[1,0,0,0,0,0,0] = -0.5

F[1,0,0,0,0,0,1] = 0.5
F[0,1,0,0,0,0,1] = -0.2
F[0,0,1,0,0,0,1] = 0.1

F[0,0,1,0,0,0,2] = -0.3
F[1,1,0,0,0,0,2] = -0.05

F[0,0,0,1,0,0,3] = 0.4
F[0,0,0,0,1,1,3] = -0.1

F[0,0,0,0,1,0,4] = -0.2
F[1,0,0,1,0,0,4] = 0.1

e = 1
F_shape = (2,)*n + (e,)
G = np.zeros(F_shape)
G[0,0,0,0,1,0,0] = 1
G[0,0,0,0,0,1,0] = -1
#We now consider the Algebraic part

F_CP_tensor = tl.decomposition.parafac(tl.tensor(F), rank=5) 
G_CP_tensor = tl.decomposition.parafac(tl.tensor(G), rank=5) 

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


z_0 = np.array([x_0[4]])
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

for i in [3,4]:
    plt.plot(t, sol[:, i], color_codes[0], label= f'y{i} EDO')
    plt.plot(t, x_sol[:, i], linestyle=':', color= color_codes[1], label=f'y{i} DAE')
    #plt.plot(t, x_solbis[:, i], linestyle='--', color= color_codes[2], label=f'y{i} DAE + Jacob')
    print("diff", sum(np.abs(sol[:, i]-x_solbis[:,i])))
    print("diff", sum(np.abs(x_sol[:, i]-x_solbis[:,i])))
    plt.show()

plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
