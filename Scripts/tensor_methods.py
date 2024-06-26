import numpy as np
import tensorly as tl
import methods_linearize as l_methods
import itertools as it
#import tensorflow as tf
import copy
from scipy.optimize import fsolve

def CP_MTI_product(CP_tensor, x):
    F_factors = CP_tensor.factors
    F_weights = CP_tensor.weights

    res_x = np.dot(F_factors[0].T,np.array([[1], [x[0]]]))*0
    res_x = np.ones(res_x.shape)
    for i in range(len(x)):
        x_i = x[i]
        F_i = F_factors[i]
        res_x = res_x*np.dot(F_i.T, np.array([[1], [x_i]]))

    
    res_x = F_factors[-1]@res_x
    res = res_x.flatten()
    return res

def MTI_product(CP_tensor, x, u =np.array([])):
    F_factors = CP_tensor.factors
    F_weights = CP_tensor.weights

    res_x = np.dot(F_factors[0].T,np.array([[1], [x[0]]]))*0
    res_x = np.ones(res_x.shape)
    xu = np.concatenate((x,u))
    for i, x_i in enumerate(xu):
        F_i = F_factors[i]
        res_x = res_x*np.dot(F_i.T,np.array([[1], [x_i]]))

    res_x = F_factors[-1]@res_x
    res = res_x.flatten()
    return res

def diffenrentiation_CP(F):
        #This functions differentiates the multilineal function defined
        #by the tensor self.F-tensor using the CP decomposition
        Theta = np.array([[0,1],[0,0]])
        D_F = []
        for (i, F_i) in enumerate(F.factors[:-1]):
            D_Fi = copy.deepcopy(F)
            D_Fi.factors[i] = Theta@F_i
            D_F.append(D_Fi)
        return D_F

def compute_diff_CP(D_F, x, u):
    n = len(D_F)
    p = D_F[0].factors[-1].shape[0]
    Jacobian = np.zeros(p, n)

    for (i, D_Fi) in enumerate(D_F):
        prod = MTI_product(D_Fi, x, u)
        Jacobian[:,i] = prod
    return Jacobian

def compute_diff_CPx(D_F, x):
    n = len(D_F)
    p = D_F[0].factors[-1].shape[0]
    Jacobian = np.zeros((p, len(x)))

    for i in range(len(x)):
        D_Fi = D_F[i]
        prod = CP_MTI_product(D_Fi, x)
        Jacobian[:,i] = prod
    return Jacobian

def inner_tensor_product(F, x):
    monomial_x = l_methods.monomial(x)
    monomial_x = tl.reshape(monomial_x , (2,)*len(x))
    axis = [i for i in range(len(x))]
    res_x = np.tensordot(F, monomial_x, axes=(axis, axis))
    return res_x

def CP_MULT_polynomialtensor(F_decomposed, x):
    #This function performs the multiplication <F| (1 x)...(1 x)>
    one_x = np.concatenate((np.array([1]),x)) 
    F_factors = F_decomposed.factors 
    F_weigths = F_decomposed.weights
    rank = F_factors[0].shape[1]
    m = F_factors[0].shape[1]
    res = np.ones(m)
    
    
    for i, fi in enumerate(F_factors[:-1]):
        res = res*np.dot(fi.T, one_x)
        
    res = np.dot(F_factors[-1], res)
    return res

def MULT_polynomialtensor(F, x):
    #This function performs the multiplication <F| (1 x)...(1 x)>
    one_x = np.concatenate((np.array([1]),x)) 
    order_F = len(F.shape)
    q = order_F-1
    m = F.shape[-1]
    res = np.ones(m)
    
    axis = list(i for i in range(order_F - 1))
    
    x_times_x = one_x
    
    for _ in range(q-1):
        x_times_x = np.tensordot(x_times_x, one_x, axes = 0)
    
    res = np.tensordot(F, x_times_x, axes =(axis, axis))
    return res

def symmetrize_tensor(tensor):
    order = tensor.ndim
    symmetrized_tensor = np.zeros_like(tensor)
    permutations = list(it.permutations(range(order)))
    for perm in permutations:
        symmetrized_tensor += np.transpose(tensor, perm)
    symmetrized_tensor /= len(permutations)
    return symmetrized_tensor

def initialize_absvar(x):
    n = len(x)
    z = np.zeros(3*len(x))
    
    for i in range(n):
        z[i] = np.abs(x[i])
        z[i+1] = max(0, x[i])
        z[i+2] = min(0, x[i])
    return z