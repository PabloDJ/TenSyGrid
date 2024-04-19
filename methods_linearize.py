import numpy as np
import tensorly as tl
from itertools import product
import copy
from scipy.optimize import fsolve
from scipy import integrate

def multilinear_base(n, order):
    base = []
    for i in product(range(2), repeat=n):
        bi = np.array(i)
        if sum(i) <= order:
            base.append(np.array(i))
    return base

def inner_product(u, v, bounds):
    bounds_var= [tuple(bound) for bound in bounds]
    def f(u, v):
        def prod(x):
            return v(x)*u(x)
        return prod
    result, error = integrate.nquad(f(u,v), bounds_var)
    return result

def graham_schmidt(base, inner_product):
    graham_base = []
    for (i,bi) in enumerate(base):
        v_save = base[i]
        for j in range(i):
            ratio = inner_product(base[v_save],base[j])/inner_product(base[v_save],base[j])
            v  = v - ratio*base[j]
        # Normalize the resulting vector and store it
        graham_base.append(v / inner_product(v,v))
    return graham_base

def tensorize(x):
    n = int(np.log2(len(x)))
    np.reshape(x, (2,) * n)
    return tl.tensor(x)

def monomial(x):
    res = tl.tensor(np.array([1]))
    for x_i in x:
        res = tl.kron(res, tl.tensor(np.array([1, x_i])))
    
    