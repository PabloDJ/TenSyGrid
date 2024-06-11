import numpy as np
import tensorly as tl
import scipy.linalg
import tensorly.contrib.sparse.decomposition as tl_sparse
from scipy.integrate import odeint
from itertools import product
from scipy.optimize import root
from scipy.sparse import coo_array
import matplotlib.pyplot as plt
#import tensorflow as tf
import MTI
import Model_MTI
import tensor_methods as methods
import methods_linearize as l_methods
import numerical as num
import time
import sparse
import itertools as it
import optimisation as optim
import tensorly.contrib.sparse as tlsp
import numerical
from scipy.optimize import approx_fprime
#np.random.seed(42) 
#We define grid parameters
L = 50
R = 0.5
C = 20

verbose = True
T = 100
#Here x = (i i'), u = (v v')
sorted(i for i in set(dir(tl)).intersection(dir(tlsp))
       if not i.startswith('_'))
print(tl.get_backend())
n_nonbin = 80
n_abs = 2
n_abstotal = n_abs*3
n = n_abstotal + n_nonbin

#We define a non linear (neither Multi Linear) time indepedenment EDO system
#We first set up the tensor related to the problem
#And we add auxiliary variable v s.t y5 = v
e = n
order = 2
F_shape = (1+n,)*order + (n_nonbin,)
F = np.zeros(F_shape)
F = np.random.rand(*F_shape) - 0.5*np.ones(F_shape)   
#F_lim_indices = (slice(1+n),)*order + (slice(e),)
#F_lim = F[F_lim_indices]
F_lim = F

def f_EDO(y, t):
    dydt = np.zeros(n)
    dydt = methods.MULT_polynomialtensor(F_lim, y)
    dydt = np.concatenate([dydt, np.random.rand(n_abstotal)])
    return dydt
                
#We now consider the Algebraic part by defining the apropriate tensor G
e = n
e = 0
G_shape = (n+1,)*order + (n_abstotal,)
G = np.zeros(G_shape)

#Not necessary given that F takes the quadric parameters into account.
for i in range(n_abs):
    n_baseline = n_nonbin+1
    G[n_baseline+3*i, n_baseline+3*i, 3*i] = 1
    G[i+1, i+1, i] = 1
    
    G[n_baseline+(3*i+1), n_baseline+(3*i+2), 3*i+1] = 1

    G[i, 0, 3*i+2] = -1
    G[n_baseline+(3*i+1), 0, 3*i+2] = 1
    G[n_baseline+(3*i+2), 0, 3*i+2] = 1
    print(f"indices {3*i}, {3*i+1}, {3*i+2}")
    
#WE CHOSE ONE OF THE FOLLOWING DECOMPOSITION METHODS
#F = methods.symmetrize_tensor(F)
#G = methods.symmetrize_tensor(G)


F_sym = np.concatenate((F, G), axis=order)
F_sym = methods.symmetrize_tensor(F_sym[:n,:n,:n])
for i,j in it.product(range(n), repeat=2):
    if F_sym[i,j,1] != F_sym[j,i,1]:
        print("non symetric") 

time_start = time.time()
F_sym_weights, F_sym_factors = tl.decomposition.symmetric_parafac_power_iteration(F_sym, rank= 80)
F_symcp_tensor = (F_sym_weights, [F_sym_factors]*len(F_sym.shape))
Sym_time = time.time() - time_start
 
time_start = time.time()
F_CP_symmetric2 = tl.decomposition.parafac(F_sym, rank= 80) 
Normal_time = time.time() - time_start

F_sym_reconstructed = tl.cp_to_tensor(F_symcp_tensor) 
F_sym_reconstructed2 = tl.cp_to_tensor(F_CP_symmetric2) 
error_sym = tl.tenalg.inner(F_sym -F_sym_reconstructed, F_sym -F_sym_reconstructed)
error_normal = tl.tenalg.inner(F_sym -F_sym_reconstructed2, F_sym -F_sym_reconstructed2)

print(f"sym decomp time is {Sym_time}")
print(f"normal decomp time is {Normal_time}")
print(f"error sym is {error_sym}")
print(f"error normal is {error_normal}")
F_CP_tensor = tl.decomposition.parafac(tl.tensor(F), rank= 25) 
G_CP_tensor = tl.decomposition.parafac(tl.tensor(G), rank= 25) 

#F_CP_tensor = tl.decomposition.symmetric_parafac_power_iteration(tl.tensor(F), rank= 25) 
try:
    G_CP_tensor = tl.decomposition.parafac(tl.tensor(G), rank=25) 
except:
    a = 0
    print("G decompositon failed")
#IF NOT PARAFAC UNCOMMENT THE FOLLOWING 2 LINES
#F_CP_tensor = MTI.Tensor_decomposition(F_CP_tensor[1],F_CP_tensor[0])
#G_CP_tensor = MTI.Tensor_decomposition(G_CP_tensor[1],G_CP_tensor[0])

eMTI = MTI.eMTI(n, 0, 0, F, G)
axis = [i for i in range(2*n)]
f1 = lambda x: methods.MULT_polynomialtensor(F, x)
f1bis = lambda x: methods.MULT_polynomialtensor(G, x)
f2 = lambda x: methods.CP_MULT_polynomialtensor(F_CP_tensor, x)
f2bis = lambda x: methods.CP_MULT_polynomialtensor(G_CP_tensor, x)
f3 = lambda x: f_EDO(x[:n], 1)

x_0 = np.random.rand(n_nonbin)
z_0 = methods.initialize_absvar(x_0[: n_abs])
xz_0 = np.concatenate((x_0, z_0))
u = np.random.rand(110,110)
dt = 0.1
t = np.linspace(0, 0.2, 100)
f1_eval = f1(xz_0)
f1bis_eval = f1bis(xz_0)
f2_eval = f2(xz_0)
f2bis_eval = f2bis(xz_0)
f3_eval = f3(xz_0)
print("xz0 is", xz_0)
print(f"f1 eval {f1_eval}")
print(f"f1bis eval {f1bis_eval}")
print(f"f2 eval {f2_eval}")
print(f"f2bis eval {f2bis_eval}")
print(f"f3 eval {f3_eval}")

EDO_start = time.time()
sol = odeint(f_EDO, xz_0, t)
EDO_time = time.time() - EDO_start
sol_trap = num.trapezoidal_implicit(f_EDO, xz_0, t)

#WE have system st x = (xdot1, x1, u1, u2, v1, v2) dxdt = (x)'
def DAE_f(x, z, t):
    xaux = np.concatenate((x,z))
    res = methods.CP_MULT_polynomialtensor(F_CP_tensor, xaux)
    return res

def DAE_f_nonCP(x, z, t):
    xaux = np.concatenate((x,z))
    res = methods.MULT_polynomialtensor(F, xaux)
    return res

def DAE_g(x, z, t):
    xaux = np.concatenate((x,z))
    try:
        res = methods.CP_MULT_polynomialtensor(G_CP_tensor, xaux)
    except:
        res = np.array([])
    #print("x  is", x[n_nonbin:])
    #print("g x is", res)
    return res

def DAE_g_nonCP(x, z, t):
    xaux = np.concatenate((x,z))
    try:
        res = methods.MULT_polynomialtensor(G, xaux)
    except:
        res = np.array([])
    #We add the constraint  
    return res

def z_explicit(x, t):
    res = np.zeros(n_abstotal)
    for i in range(n_abs):
        v = x[i]
        res[3*i + 0] = np.abs(v) 
        res[3*i + 1] = max(v, 0) 
        res[3*i + 2] = min(v, 0)
    return res 

for i in range(100):
    break
    x_0 = 10*np.random.rand(n_nonbin)
    z_0 = methods.initialize_absvar(x_0[: n_abs])
    z_0 = 10*np.random.rand(len(z_0))
    xz_0 = np.concatenate((x_0, z_0))
    def func(xz):
        empty_array = np.array([])
        res1 = DAE_f(xz, empty_array, 1)
        res2 = DAE_g(xz, empty_array, 1)
        res = np.concatenate((res1, res2))
        return res
    J = numerical.jacobian(func, xz_0)
    J1 = numerical.jacobian(func, xz_0)[:n_nonbin, :n_nonbin]
    J2 = numerical.jacobian(func, xz_0)[n_nonbin:-2, n_nonbin:-2]
    rank0 = np.linalg.matrix_rank(J)
    det0 = np.linalg.det(J)
    rank1 = np.linalg.matrix_rank(J1)
    det1 = np.linalg.det(J1)
    rank2 = np.linalg.matrix_rank(J2)
    det2 = np.linalg.det(J2)
    print(f"for J0 ranke is {rank0} and det is {det0}")
    print(f"for J1 ranke is {rank1} and det is {det1}")
    print(f"for J2 ranke is {rank2} and det is {det2}")
#DF = methods.diffenrentiation_CP(F_CP_tensor)
#DG = methods.diffenrentiation_CP(G_CP_tensor)

def DAE_J(x, z, t):
    xaux = np.concatenate((x,z))
    partialx_F = methods.compute_diff_CPx(DF, xaux)
    partialx_G = methods.compute_diff_CPx(DG, xaux)
    J = np.concatenate((partialx_F, partialx_G))
    return J

DAE_start = time.time()
#x_sol, z_sol = num.backward_euler_semi_explicit(DAE_f, DAE_g, x_0, z_0, t)
DAE_time = time.time()- DAE_start
DAE2_start = time.time()
x_solbis, z_solbis = num.backward_euler_semi_explicit(DAE_f, DAE_g, x_0, z_0, t, DAE_Jacob = False)
DAE2_time = time.time()- DAE2_start
DAE3_start = time.time()
x_solbis2, z_solbis2 = num.algebraic_substitution_semi_explicit(DAE_f, DAE_g, x_0, z_0, t, z_explicit =z_explicit)
DAE3_time = time.time()- DAE3_start
x_sol, z_sol = sol, 0
x_solbis, z_solbis = sol, 0
DAE3_time = time.time() - DAE3_start

color_codes = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
print(f"time EDO is {EDO_time}")
print(f"time DAE is {DAE_time}")
print(f"time DAE 2 is {DAE2_time}")
print(f"time DAE algebraic is {DAE3_time}")
print(x_solbis2.shape)

for i in range(2):
    plt.plot(t, sol[:, i], color_codes[0], linestyle=':', label= f'y{i} EDO')
    plt.plot(t, sol[:,n_nonbin + 3*i], color_codes[1], linestyle=':', label= f'y abs {i} EDO')
    plt.plot(t, sol[:,n_nonbin + 3*i+1], color_codes[2], linestyle=':', label= f'y^+{i} EDO')
    plt.plot(t, sol[:,n_nonbin + 3*i+2], color_codes[3], linestyle=':', label= f'y^-{i} EDO')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.show()
    
    #plt.plot(t, x_sol[:, n_nonbin + i], linestyle=':', color= color_codes[1], label=f'y{i} DAE')
    plt.plot(t, x_solbis2[:, i], linestyle=':', color= color_codes[0], label=f'y{i} DAE 2')
    plt.plot(t, z_solbis2[:, 3*i+0], linestyle=':', color= color_codes[1], label=f'y abs {i} DAE 2')
    plt.plot(t, z_solbis2[:, 3*i+1], linestyle=':', color= color_codes[2], label=f'y +{i} DAE 2')
    plt.plot(t, z_solbis2[:, 3*i+2], linestyle=':', color= color_codes[3], label=f'y -{i} DAE 2')
    print("diff EDO DAE", sum(np.abs(sol[:, i]-x_solbis[:,i])))
    print("diff DAE DAE", sum(np.abs(x_sol[:, i]-x_solbis[:,i])))
    print("diff DAE ALG", sum(np.abs(x_sol[:, i]-x_solbis2[:,i])))
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    plt.show()

verbose = False
if verbose==True:
    for ti in range(len(t)):
        print("timestep is ", ti)
        print(f"x EDO {sol[ti,:]}")
        print(f"dxdt EDO {f_EDO(sol[ti,:],t)}")
        print(f"x DAE {x_sol[ti,:]}")
        print(f"f EDO DAE {f_EDO(x_sol[ti,:],t)}")
        print(f"f1 DAE {f1(np.concatenate((x_sol[ti,:], x_sol[ti,:])))}")
