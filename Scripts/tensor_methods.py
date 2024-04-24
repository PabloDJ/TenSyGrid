import numpy as np
import tensorly as tl
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
        res_x = res_x*np.dot(F_i.T,np.array([[1], [x_i]]))

    res_x = F_factors[-1]@res_x
    res = res_x.flatten()
    return res

def MTI_product(CP_tensor, x, u):
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
