import numpy as np
import tensorly as tl
from scipy.integrate import odeint
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import root
#import tensorflow as tf
import MTI
import time
import tensor_methods


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
p = 0
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

#We set up the first equation (i)' = i'
F[0,1,0,0, 0] = 1

#Second Equation i'' = (-1/C) *i - Ri' +v'
F[1,0,0,0, 1] = -1/C
F[0,1,0,0, 1] = -R
F[0,0,0,1, 1] = 1
F = np.random.rand(*F.shape)

eMTI = MTI.eMTI(n, m, p, F, G)
iMTI = MTI.iMTI(n, m, p, n+m, F)
eMTI.CP_decomposition(rank = 10)
print(eMTI.F_factors)

#We define the ODE parameter
T = 100
dt = 0.02

#Then the voltage derivative v'
v0 = 10
dvdt = 100*np.ones(T)
v = v0*np.ones(T) + np.cumsum(dvdt*dt)
u = np.array([v, dvdt])

#Then the current
didt2_0 = 1 
didt_0 = C*(dvdt[0]-didt2_0)
i_0 = didt_0
x_0 = np.array([i_0, didt_0])

compact = True
t = np.linspace(0, (T - 1) * dt, T)
f1 = eMTI.CP_stepforward(compact)
f2 = eMTI.tensor_stepforward(compact)


f1_eval = f1(x_0, 0, u, dt)
f2_eval = f2(x_0, 0, u, dt)
print(f"f1 eval {f1_eval}")
print(f"f2 eval {f2_eval}")



raise Exception("Stopping script here")
B = np.array([[0,0],[0,1]])
C = np.array([[0,1],[-1/C,-R]])
res = np.dot(C,x_0) + np.dot(B,u[:,0])
for x0, x1, u0, u1 in product(range(2), repeat=4):
    x_0 = np.array([x0, x1])
    u[:,0] = np.array([u0, u1])
    f1_eval = f1(x_0, 0, u, dt)
    f2_eval = f2(x_0, 0, u, dt)
    res = np.dot(C,x_0) + np.dot(B,u[:,0])
    print((x0, x1, u0, u1), " ", f"f1 eval {f1_eval} f2 eval {f2_eval} res {res}")

#raise Exception("Stopping script here")
CP_ODE_start = time.time()
sol = odeint(eMTI.CP_stepforward(compact), x_0, t, args=(u, dt))
CP_ODE_end = time.time()
CP_ODE_time = CP_ODE_end - CP_ODE_start
print(f"Computation time CP= {CP_ODE_time}")

plt.plot(t, sol[:, 0], 'b', label='I(t)')
plt.plot(t, sol[:, 1], 'g', label='dI/dt(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

tensor_ODE_start = time.time()
sol = odeint(eMTI.tensor_stepforward(compact), x_0, t, args=(u, dt))
tensor_ODE_end = time.time()
tensor_ODE_time = tensor_ODE_end - tensor_ODE_start
print(f"Computation time with tensors = {tensor_ODE_time}")

plt.plot(t, sol[:, 0], 'b', label='I(t)')
plt.plot(t, sol[:, 1], 'g', label='dI/dt(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()