import numpy as np
import tensorly as tl
from scipy.integrate import odeint
from itertools import product
from scipy.optimize import root
import matplotlib.pyplot as plt
#import tensorflow as tf
import MTI
import tensor_methods as methods

order_N = 6

x_max = 10*np.ones(4)
x_min = np.zeros(4)