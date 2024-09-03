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
import sympy as sp
import andes_methods as ad_methods
import os
import pickle
from scipy.optimize import approx_fprime

if __name__ == '__main__':
    import andes
    from andes.utils.paths import get_case, cases_root, list_cases

andes.config_logger(30)
list_cases()

sys = andes.run(get_case('kundur/kundur_full.xlsx'))
sys.PFlow.run()
sys.TDS.run()

F_sym = ad_methods.load_variable('F_sym.pkl')
G_sym = ad_methods.load_variable('G_sym.pkl')

if F_sym is None or G_sym is None:
    F_sym, G_sym = ad_methods.sys_to_eq(sys, bool_print = False)
    ad_methods.save_variable(F_sym, 'F_sym.pkl')
    ad_methods.save_variable(G_sym, 'G_sym.pkl')
    
F_poly = ad_methods.equations_to_poly(F_sym)
G_poly = ad_methods.equations_to_poly(G_sym)

F_tensor = ad_methods.poly_to_tensor(F_poly)
G_tensor = ad_methods.poly_to_tensor(F_poly)

raise Exception("stop")
#sym_processor = sys.syms  e
atributes = vars(sys)
atributes = vars(bus)
verbose = False

ss = andes.System()
ss.GENCLS.prepare()

if verbose:
    for name, model in models.items():
        print("name in model is", name, model)

    for name, model in atributes.items():
        print("name in atribute is", name, model)

    ad_methods.numerical_system(sys)
    print("getting dae attributes")
    ad_methods.get_attributes(dae)
    ad_methods.get_attributes(bus)
    
    print("bus variable is ", bus.wd)

groups = sys.groups
for group in groups:
    ad_methods.get_attributes(group)


vars = ss.GENCLS.syms.xy
equations = ss.GENCLS.syms.f

ad_methods.symbolic_to_tensor(vars, equations)
#raise Exception("stop")
           
