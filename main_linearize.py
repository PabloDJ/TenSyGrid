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

order = 6
n = 6

x_max = 10*np.ones(4)
x_min = np.zeros(4)
bounds = np.array([x_min, x_max])

def inner_product(u,v):
    def u_aux(x):
        monomial_x = l_methods.monomial(x)
        u_tensor = l_methods.tensorize(u)
        res = tl.dot(u_tensor, monomial_x)
        return res
    def v_aux(x):
        monomial_x = l_methods.monomial(x)
        v_tensor = l_methods.tensorize(v)
        res = tl.dot(v_tensor, monomial_x)
        return res
    return l_methods.inner_product(u_aux, v_aux, bounds)

base = l_methods.multilinear_base(6)
base_GS = l_methods.graham_schmidt(base, inner_product)

