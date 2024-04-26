import numpy as np
import tensorly as tl
from scipy.integrate import odeint
from itertools import product
from scipy.optimize import root
import matplotlib.pyplot as plt
#import tensorflow as tf
import MTI
import tensor_methods as t_methods
import methods_linearize as l_methods
import sympy as sp

order = 4
n = 4

x_max = 1*np.ones(n)
x_min = 0*np.zeros(n)
bounds = np.array([x_min, x_max])
bounds_var = [[bounds[0,i], bounds[1,i]] for i in range(len(bounds[0]))]

#ALTERNATIVAMENTE DESCOMPONER U Y V Y USAR MULTIPLICACION cp
#Pero hay que descomponer a cada par
#u(x)*v(x) es polinomico

def inner_product_symbolic(u,v):
    expr_u, x = l_methods.tensor_to_simbolic(u)
    expr_v, x = l_methods.tensor_to_simbolic(v)
    expr_mult = expr_u*expr_v
    #to avoid very small coeficcients
    #expr_mult = sp.simplify(expr_u*expr_v)
    return l_methods.integrate_symbolic_defined(x, expr_mult, bounds_var)

def inner_product_CP(u,v):
    n = len(u.shape)
    new_shape = u.shape + (1,)
    u = np.reshape(u, new_shape)
    v = np.reshape(v, new_shape)
    v2 = np.zeros(new_shape)
    for vi in np.ndindex(v.shape):
        v2[vi] = float(v[vi])
    def u_aux(x):
        u2 = tl.decomposition.parafac(u, rank=1)
        res = t_methods.CP_MTI_product(u2, x)
        return res
    def v_aux(x):
        v2_decom = tl.decomposition.parafac(v2, rank=1)
        res = t_methods.CP_MTI_product(v2_decom, x)
        return res
    res = l_methods.inner_product(u_aux, v_aux, bounds)
    return res


def inner_product_hybrid(u,v):
    expr_u, x = l_methods.tensor_to_simbolic(u)
    expr_v, x = l_methods.tensor_to_simbolic(v)
    expr_mult = sp.simplify(expr_u*expr_v)
    return l_methods.integrate_symbolic_defined(x, expr_mult, bounds_var)

base = l_methods.multilinear_base(n, order = 2)
tensor_base = [l_methods.tensorize(bi) for bi in base]
base_GS = l_methods.graham_schmidt(tensor_base, inner_product_symbolic)

for i in range(len(base_GS)):
    for j in range(i):
        print("product of ", i, j, " ",inner_product_symbolic(base_GS[j], base_GS[i]))

array_shape = (2,)*n

def F(x):
    A = np.random.rand(n)
    return x.T@A@x 

F = np.random.rand(*array_shape)
coef = np.zeros(len(base_GS))

for i in range(len(coef)):
    coef[i] = inner_product_symbolic(F, base_GS[i])

tensor_aproximation = np.zeros(base_GS[0].shape)
for (i,c) in enumerate(coef):
    tensor_aproximation = c*base_GS[i]

print("error is", np.sum(np.abs(tensor_aproximation - F)))
for i in np.ndindex(F.shape):
    print("error in ", i," is ", np.abs(tensor_aproximation[i] - F[i]))