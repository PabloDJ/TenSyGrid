import numpy as np
import tensorly as tl
from itertools import product
import copy
from scipy.optimize import fsolve

def multilinear_base(n, order):
    base = []
    for i in product(range(2), repeat=n):
        bi = np.array(i)
        if sum(i) <= order:
            base.append(np.array(i))
    return base

def graham_schmidt(base, inner_product):
    graham_base = []
    for (i,bi) in enumerate(base):
        v = A[i]
        for j in range(i):
            v -= np.dot(Q[j], A[i]) * Q[j]
        # Normalize the resulting vector and store it
        vi = v / inner_product(v,v)