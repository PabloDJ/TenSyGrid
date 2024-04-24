import numpy as np
import tensorly as tl
from itertools import product
import sympy as sp
from scipy.optimize import fsolve
from scipy import integrate

def multilinear_base(n, order):
    base = []
    for i in product(range(2), repeat=n):
        bi = np.array(i)
        if sum(bi[:]) <= order:
            base.append(bi)
    return base

def inner_product(u, v, bounds):
    bounds_var= [[bounds[0,i], bounds[1,i]] for i in range(len(bounds[0]))]
    def f(u, v):
        def prod(x):
            return v(x)*u(x)
        return prod
    
    func = f(u,v)
    def func_wrapper(*args):
        res = func(np.array(args))
        return res
    print("integration start")
    result, error = integrate.nquad(func_wrapper, ranges=bounds_var)
    print("integration end")
    return result

def graham_schmidt(base, inner_product):
    u_base = []
    graham_base = []
    for (i,bi) in enumerate(base):
        v_save = bi
        u_next = bi
        print(f"iteration i {i}")
        for j in range(i):
            e_last = graham_base[j]
            ratio = inner_product(v_save, e_last)
            u_next  = u_next - ratio*e_last
        # Normalize the resulting vector and store it
        graham_base.append(u_next / inner_product(u_next,u_next))
    return graham_base

def inner_product_wrapper(bounds):
    def func(u,v):
        def u_aux(x):
            monomial_x = monomial(x)
            monomial_x = tl.reshape(monomial_x , (2,)*len(x))
            axis = [i for i in range(len(x))]
            res_x = np.tensordot(u, monomial_x, axes=(axis, axis))
            return res_x
        def v_aux(x):
            monomial_x = monomial(x)
            monomial_x = tl.reshape(monomial_x , (2,)*len(x))
            axis = [i for i in range(len(x))]
            res_x = np.tensordot(v, monomial_x, axes=(axis, axis))
            return res_x
        return inner_product(u_aux, v_aux, bounds)
    return func

def tensor_to_simbolic(F):
    x = []
    n = len(F.shape)
    for i in range(n):
        x_i = sp.symbols(f'x_{i}')
        x.append(x_i)

    expr = 0
    for index in np.ndindex(F.shape): 
        x_monomial = 1 
        for i in range(n):
            x_monomial *= x[i]**index[i]
        expr += F[index]*x_monomial
    return expr, x

def integrate_symbolic(x, expr):
    integrated_expr = expr
    for xi in x:
        integrated_expr = sp.integrate(integrated_expr, xi)
    return integrated_expr

def integrate_symbolic_defined(x, expr, bounds):
    integrated_expr = expr
    for (i,xi) in enumerate(x):
        integrated_expr = sp.integrate(integrated_expr, (xi,) + (bounds[i],))
    return integrated_expr

def tensorize(x):
    tensor_shape = (2,)*len(x)
    res = tl.zeros(tensor_shape)
    res[tuple(x)] = 1
    return res

def monomial(x, tensorize = False):
    res = tl.tensor(np.array([1]))
    for x_i in x:
        res = tl.kron(res, tl.tensor(np.array([1, x_i])))
    
    if tensorize:
        res = np.reshape(res, (2,)*int(np.log2(len(res))))
    return res
    
