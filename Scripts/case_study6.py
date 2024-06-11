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
import io
import re
from scipy.optimize import approx_fprime

if __name__ == '__main__':
    import andes
    from andes.utils.paths import get_case, cases_root, list_cases

andes.config_logger(30)
list_cases()

sys = andes.run(get_case('kundur/kundur_full.xlsx'))

dae = sys.dae
models = sys.models
TGV = sys.TGOV1
bus = sys.Bus
line = sys.Line
generator = sys.PQ
l1 = line.bus1.get_names()
l2 = line.bus1

a1 = sys.find_models('tds')
a2 = sys.find_models('pflow')

dev = sys.find_devices()

#we define an empty system ssand prepare the algebraic equations
ss = andes.System()
ss.GENCLS.prepare()

#we define the system sys 2
sys2 = andes.System()
sys2.Bus.add(idx=1, id='Bus1', type='slack', V=1.0, theta=0.0)
sys2.Bus.add(idx=2, id='Bus2', type='PQ', P=50.0, Q=30.0)

# Add a generator with the GENCLS model
sys2.GENCLS.add(idx=1, id='Gen1', bus=1, gen =1, P=80.0, Q=20.0, Vset=1.0)

#ad_methods.prepare_all_models(sys)
#ad_methods.prepare_all_models(sys2)
#ad_methods.e_to_dae(ss)
#ad_methods.e_to_dae(sys)
#ad_methods.e_to_dae(sys2)
#ad_methods.alt_prepare(sys)

sys.PFlow.run()
sys.TDS.run()

F_sym, G_sym = ad_methods.sys_to_eq(sys, bool_print = True)
F_poly = ad_methods.equations_to_poly(F_sym)
G_poly = ad_methods.equations_to_poly(G_sym)
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
           
