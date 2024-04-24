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
F_newshape = (2,2,2,2,2,2,2)
F_save = F
F = np.random.rand(*F_newshape)
F = np.zeros(F_newshape)
F[:, :, :, :, 0:2] = F_save
F = np.random.rand(*F_newshape)


eMTI = MTI.eMTI(n, m, p, F, G)
iMTI = MTI.iMTI(n, m, p, e, F)
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

def func(F_CP):
    def res(x_save):
        x = x_save[0:n]
        u = x_save[n:n+m]
        return methods.MTI_product(F_CP, x, u)
    return res

def jacob(F_CP):
    DF = methods.diffenrentiation_CP(0, F_CP)
    def res(x_save):
        res = []
        x = x_save[0:n]
        u = x_save[n:n+m]
        for (i, D_Fi) in enumerate(DF):
            res.append(methods.MTI_product(D_Fi, x, u))
        return np.array(res).T
    return res 

x0 = np.array([i_0, didt_0])
u_0 = np.array([0,0])
u = np.zeros((T,2))
u = np.random.rand(T,2)
iMTI.set_MTI_state(np.array([i_0, didt_0]))
iMTI.set_MTI_control(u_0)
iMTI.CP_decomposition(rank = 10)

model_mti = Model_MTI.Model_MTI(iMTI, x0, u, t, dt, T)
model_mti.update_MTI()
model_mti.x = np.array(model_mti.x)
print(" sol shape ", model_mti.x.shape)

plt.plot(range(len(model_mti.x[:, 0])), model_mti.x[:, 0], 'b', label='I(t)')
plt.plot(range(len(model_mti.x[:, 1])), model_mti.x[:, 1], 'g', label='dI/dt(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

raise Exception("end scrpt")
f = iMTI.Implicit_function()
j = iMTI.Implicit_diff()

print(f(x0))
print(j(x0))
sol = root(f, x0, jac=j)
print(f" sol{sol}")
print(f" f(x*) {f(sol.x)}")
print(f" J(x*) {j(sol.x)}")