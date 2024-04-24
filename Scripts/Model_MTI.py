import numpy as np
import MTI
import scipy
from scipy.optimize import root

class Model_MTI:
    def __init__(self, mti, x0, u, t, dt, T):
        self.mti = mti
        self.x = [x0]
        self.u = u
        self.t = t
        self.T = T
        self.dt = dt
        self.y = []
        self.xdot = []

    def update_MTI(self):
        self.mti.CP_decomposition(rank = 5)
        for t in range(self.T):
            print(f"iteration: {t}")
            self.mti.set_MTI_state(self.x[-1])
            self.mti.set_MTI_control(self.u[t])
            f = self.mti.Implicit_function()
            j = self.mti.Implicit_diff()
            if len(self.xdot) != 0:
                x0 = np.array(self.xdot[-1])
                if self.mti.p != 0:
                    y_array = self.y[-1]
                    x0 = np.concatenate((x0, y_array))
            else:
                x0 = np.random.rand(self.mti.p + self.mti.m)
            sol = root(f, x0, jac=j)
            print(f"solution is {sol.x}")
            xdot = sol.x[:self.mti.n]
            y = sol.x[self.mti.n:]
            self.xdot.append(xdot)

            x_new = self.x[-1] + xdot*self.dt
            print(f"x new: ")
            self.x.append(x_new)
            self.y.append(y)

            print(" state update is ",self.x[-1])
            
            
