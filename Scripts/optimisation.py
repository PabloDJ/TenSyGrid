import numpy as np
import tensorly as tl
import methods_linearize as l_methods
import itertools as it
#import tensorflow as tf
from ortools.sat.python import cp_model
import copy
from scipy.optimize import fsolve

def multiply_CP(G_factors, G_last, index, rank, n):
    i,j,l = index
    G_factors
    res = 0
    for r in range(rank):
        res_rank = 1
        for k in range(n):
            if i == k or i ==j:
                res_rank *= (G_factors[i][1,r])
            else:
                res_rank *= (G_factors[i][0,r])
        res_rank *= (G_last[l,r])
        res += res_rank                
    return res

def multiply_CP(G_factors, G_last, index, rank, n):
    i,j,l = index
    G_factors
    res = 0
    for r in range(rank):
        res_rank = 1
        for k in range(n):
            if i == k or i ==j:
                res_rank *= (G_factors[i][1,r])
            else:
                res_rank *= (G_factors[i][0,r])
        res_rank *= (G_last[l,r])
        res += res_rank                
    return res

"""def cvxpy_problem(F, F_factors, order = 2):
    r = F_factors[0].shape[1]
    n = F.shape[0]
    p = F.shape[-1]
    G_factors = [cp.Variable((2, r)) for i in range(n)]
    G_last = cp.Variable((p, r))
    obj = 0
    
    for i,j,l in it.product(range(n),range(n),range(p)):
        index = (i,j,l)
        G_value = multiply_CP(G_factors, G_last, index, r, n)
        F_value = F[i,j,l]
        obj += cp.abs(G_value - F_value)
        
    cp_problem = cp.Problem(cp.Minimize(obj), [])
    cp_problem.solve()
    
    return G_factors.value, G_last.value"""

def modified_CP_ALS(F):
    
    return 0
    