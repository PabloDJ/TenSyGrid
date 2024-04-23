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

order = 2
n = 4

x_max = 10*np.ones(4)
x_min = 9*np.zeros(4)
bounds = np.array([x_min, x_max])

#ALTERNATIVAMENTE DESCOMPONER U Y V Y USAR MULTIPLICACION cp
#Pero hay que descomponer a cada par
#u(x)*v(x) es polinomico

def inner_product(u,v):
    def u_aux(x):
        monomial_x = l_methods.monomial(x)
        monomial_x = tl.reshape(monomial_x , (2,)*len(x))
        axis = [i for i in range(len(x))]
        res_x = np.tensordot(u, monomial_x, axes=(axis, axis))
        return res_x
    def v_aux(x):
        monomial_x = l_methods.monomial(x)
        monomial_x = tl.reshape(monomial_x , (2,)*len(x))
        axis = [i for i in range(len(x))]
        res_x = np.tensordot(v, monomial_x, axes=(axis, axis))
        return res_x
    return l_methods.inner_product(u_aux, v_aux, bounds)


base = l_methods.multilinear_base(n, order = 2)
tensor_base = [l_methods.tensorize(bi) for bi in base]
expr, x = l_methods.tensor_to_simbolic(tensor_base[1])
expr = l_methods.integrate_symbolic(x, expr)
print(expr)
raise Exception("stop")
print("IP of", inner_product(tensor_base[0], tensor_base[1]) )
base_GS = l_methods.graham_schmidt(tensor_base, inner_product)

