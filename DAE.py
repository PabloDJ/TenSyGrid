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
#We define a non linear EDO system
def f_EDO(x, t):
    x1 = x[0]
    x2 = x[1]
    res = 0*x
    delta = np.pi/3
    res[0] = x1*np.cos(x2 + delta)
    res[1] = x1*np.sin(x2 + delta)
    return res

x_0 = np.random.rand(2)
sol = odeint(f_EDO, x_0, t)

plt.plot(t, sol[:, 0], 'b', label='I(t)')
plt.plot(t, sol[:, 1], 'g', label='dI/dt(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
#plt.show()

#Now we define the DAE
e = 5
F = np.zeros( (2,)*6 + (e,))    
F[0, 1, 1, 0, 0, 0, 0] = 1
F[1, 0, 0, 0, 0, 0, 0] = -1

F[0, 1,  0, 1, 0, 0, 1] = 1
F[1, 0,  0, 0, 0, 0, 1] = -1

F[0, 0,  1, 0, 0, 0, 2] = 1
F[0, 0,  0, 0, 1, 0, 2] = -1

F[0, 0,  0, 1, 0, 0, 3] = 1
F[0, 0,  0, 0, 0, 1, 3] = -1

F[0, 0,  0, 1, 0, 1, 4] = 1
F[0, 0,  1, 0, 1, 0, 4] = 1
F[0, 0,  0, 0, 0, 0, 4] = -1
F_CP_tensor = tl.decomposition.parafac(tl.tensor(F), rank=1) 

#WE have system st x = (x1, u1, u2, v1, v2) x' = (x)
def DAE_f(x_aux, dxdt, t):
    print(f"xaux {x_aux}")
    print(f"dxdt[0:2] {dxdt[0:2]}")
    x = np.zeros(len(x_aux) + 1)
    x[0:1] = dxdt[0:1]
    x[1:] = x_aux
    return methods.CP_MTI_product(F_CP_tensor, x) 

def DAE_J(x_aux, dxdt, t):
    DF = methods.diffenrentiation_CP(F_CP_tensor)
    x = np.zeros(len(x_aux) + 1)
    x[0:1] = dxdt[0:1]
    x[1:] = x_aux
    print("x shape", x.shape)
    return methods.compute_diff_CPx(DF, x)

x1 = np.random.rand()
x2 = np.random.rand()
u1 = np.cos(x2)
u2 = 1 - u1**2
x_0 = np.array([x1, u1, u2, u1, u2])
sol = num.backward_euler_DAE(DAE_f, DAE_J, x_0, t)

plt.plot(t, sol[:, 0], 'b', label='I(t)')
plt.plot(t, sol[:, 1], 'g', label='dI/dt(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()





