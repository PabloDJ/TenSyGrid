import numpy as np
import tensorly as tl
import MTI

#We define grid parameters
L = 5
R = 0.5
C = 2

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
G_size = (2 for _ in range(2**(n+m))) 
G_size = tuple(G_size) + (p,)
F_size = (2 for _ in range(2**(n+m))) 
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
print()