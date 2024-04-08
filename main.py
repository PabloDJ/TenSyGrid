import numpy as np
import tensorly as tl
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import MTI

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


eMTI = MTI.eMTI(n, m, p, F, G)
eMTI.CP_decomposition()
print(eMTI.F_factors)

#We define the ODE parameter
T = 100
dt = 0.02

#Then the current 
i_0 = 1
didt_0 = 0
x_0 = np.array([i_0, didt_0])

#Then the voltage derivative v'
v0 = 200
dvdt = 100*np.random.rand(T)
v = v0*np.ones(T) + np.cumsum(dvdt*dt)
u = np.array([v, dvdt])

compact = True
t = np.linspace(0, (T - 1) * dt, T)
print(t)
sol = odeint(eMTI.CP_stepforward(), x_0, t, args=(compact, u, dt))

plt.plot(t, sol[:, 0], 'b', label='I(t)')
plt.plot(t, sol[:, 1], 'g', label='dI/dt(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()