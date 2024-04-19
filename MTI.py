import numpy as np
import copy
from itertools import product
import tensorly as tl
import tensor_methods as tmethods

class eMTI:
    def __init__(self, n, m, p, F, G):
        self.n = n
        self.m = m
        self.p = p
        self.F = F
        self.G = G
    
    def CP_decomposition(self, normalize= True, rank = 1):
        F_tensor = tl.tensor(self.F)
        G_tensor = tl.tensor(self.G)
        self.F_tensor = F_tensor
        try:
            self.G_tensor = G_tensor
        except:
            self.G_tensor = F_tensor
        if self.n !=0:
            self.F_factors = tl.decomposition.parafac(F_tensor, rank)
        if self.p !=0:
            print(G_tensor)
            self.G_factors = tl.decomposition.parafac(G_tensor, rank)

    def CP_stepforward(self, compact=True):

        def definition(x, t, u, dt, compact = compact, verbose = False):
            T = len(u[:,0])
            u = u[:, min(T-1, round(t/dt))]

            res_x = np.array([])
            res_y = np.array([])
            F_factors = []
            F_weights = []
            G_factors = []
            G_weights = []

            if self.n!= 0:
                F_factors = self.F_factors.factors
                F_weights = self.F_factors.weights
                res_x = np.dot(F_factors[0].T,np.array([[1], [x[0]]]))*0
                res_x = np.ones(res_x.shape)
            if self.p!=0:
                G_factors = self.G_factors.factors
                G_weights = self.G_factors.weights
                res_y = np.dot(F_factors[0].T,np.array([[1], [u[0]]]))*0
                res_y = np.ones(res_y.shape)

            xu = np.concatenate((x,u))
            for i, x_i in enumerate(xu):
                F_i = F_factors[i]
                try:
                    G_i = G_weights[i]*G_factors[i]
                except:
                    G_i = F_factors[i]
                res_x = res_x*np.dot(F_i.T,np.array([[1], [x_i]]))
                #res_y = res_y*np.dot(G_i.T,np.array([[1, x_i]]))
                if verbose:
                    print(f"i {i}, xi {x_i}, F_i {F_i.T}, factor {np.dot(F_i.T,np.array([1, x_i]))}, resx {res_x}" )
            
            res_x = F_factors[-1]@res_x
            res = (res_x, res_y)

            if compact:
                res = res_x.flatten()
            return res
        return definition
    
    def CPN_stepforward(self, compact=True):

        def definition(x, t, u, dt, compact = compact, verbose = False):
            T = len(u[:,0])
            u = u[:, min(T-1, round(t/dt))]

            res_x = np.array([])
            res_y = np.array([])
            F_factors = []
            F_weights = []
            G_factors = []
            G_weights = []

            if self.n!= 0:
                F_factors = self.F_factors.factors
                F_weights = self.F_factors.weights
                res_x = np.dot(F_factors[0].T,np.array([[1], [x[0]]]))*0
                res_x = np.ones(res_x.shape)
            if self.p!=0:
                G_factors = self.G_factors.factors
                G_weights = self.G_factors.weights
                res_y = np.dot(F_factors[0].T,np.array([[1], [u[0]]]))*0
                res_y = np.ones(res_y.shape)

            xu = np.concatenate((x,u))
            for i, x_i in enumerate(xu):
                F_i = F_factors[i]
                try:
                    G_i = G_weights[i]*G_factors[i]
                except:
                    G_i = F_factors[i]
                res_x = res_x*np.dot(F_i.T,np.array([[1], [x_i]]))
                #res_y = res_y*np.dot(G_i.T,np.array([[1, x_i]]))
                if verbose:
                    print(f"i {i}, xi {x_i}, F_i {F_i.T}, factor {np.dot(F_i.T,np.array([1, x_i]))}, resx {res_x}" )
            
            res_x = F_factors[-1]@res_x
            res = (res_x, res_y)

            if compact:
                res = res_x.flatten()
            return res
        return definition
    
    def tensor_stepforward(self, compact=True):
        
        def definition(x, t, u, dt, compact=compact):
            T = len(u[:,0])
            u = u[:, min(T-1, round(t/dt))]

            m_xu = tl.tensor(np.array([1]))

            for x_i in x:
                aux = np.array([[1], [x_i]])
                m_xu = tl.kron(m_xu, aux)
            for u_i in u:
                aux = np.array([[1], [u_i]])
                m_xu = tl.kron(m_xu, aux)
            
            new_shape = tuple(2 for _ in range(self.n + self.m)) 
            m_xu = tl.reshape(m_xu, new_shape)
            
            axis = [i for i in range(self.n + self.m)]
            res_x = np.tensordot(self.F, m_xu, axes=(axis, axis))
            if compact:
                res = res_x.flatten()
            return res
        
        return definition
  
    def convert_to_iMTI(self, equality=True):
        N = self.n + self.p
        H_size = (2 for _ in range(2**(2*self.n+self.m+self.p))) + (N,)
        H = np.zeros(H_size)

        #WE FIRST EXPLICTLY DEFINE THE COEFICIENTS FOR x' and y
        for i in range(self.n):
            index = np.zeros(H_size)
            index[i] = 1 
            H[tuple(index)] = H[tuple(index)]  + 1
        for i in range(self.p):
            index = np.zeros(H_size)
            index[i + 2**(self.n+self.n+self.m)] = 1 
            H[tuple(index)] = H[tuple(index)]  + 1

        F_slice = tuple(0 for _ in range((self.n)))
        F_slice = F_slice + tuple(slice(0,2) for _ in range((self.n)))
        F_slice = F_slice + tuple(slice(0,2) for _ in range((self.m)))
        F_slice = F_slice + tuple(0 for _ in range((self.p)))
        F_slice = F_slice + (slice(0, self.n),)
        H[F_slice] = -self.F

        G_slice = tuple(0 for _ in range((self.n)))
        G_slice = G_slice + tuple(slice(0,2) for _ in range((self.n)))
        G_slice = G_slice + tuple(slice(0,2) for _ in range((self.m)))
        G_slice = G_slice + tuple(0 for _ in range((self.p)))
        G_slice = G_slice + (slice(0, self.p),)
        H[G_slice] = -self.G
        if equality:
            imti = iMTI(self.n, self.m, self.p, H) 
        else:
            L = (2 for _ in range(2**(2*self.n+self.m+self.p))) + (2*N,)
            slice1 = tuple(slice(0,2) for _ in range((self.n)))
            slice1 = slice1 + tuple(slice(0,2) for _ in range((self.n)))
            slice1 = slice1 + tuple(slice(0,2) for _ in range((self.m)))
            slice1 = slice1 + tuple(slice(0,2) for _ in range((self.p)))
            sliceH = slice1 + tuple(slice(0, self.N),)
            sliceH2 = slice1 + tuple(slice(self.N, 2*self.N))

            L[sliceH] = H
            L[sliceH2] = -H
            imti = iMTI(self.n, self.m, self.p, L)
        return imti

class iMTI:
    def __init__(self, n, m, p, e, H):
        self.n = n
        self.m = m
        self.p = p
        self.e = e
        self.H = H

    def set_MTI_state(self, x):
        self.x = copy.deepcopy(np.array(x))
    
    def set_MTI_control(self, u):
        self.u = np.array(u)

    def CP_decomposition(self, rank = 5):
        H_tensor = tl.tensor(self.H)
        self.H_factors = tl.decomposition.parafac(H_tensor,rank)
    
    def CP_multiplication(self):
        def CP_mult_aux(x):
            H_factors = self.H_factors.factors
            H_weigths = self.H_factors.weights
            res_x = np.dot(H_factors[0].T,np.array([[1], [x[0]]]))*0
            res_x = np.ones(res_x.shape)
            print(f" # of factors {len(H_factors)}")
            print(f" x shape {x}")
            for i, x_i in enumerate(x):
                H_i = H_factors[i]
                res_x = res_x*np.dot(H_i.T,np.array([[1], [x_i]]))
            res_x = H_factors[-1]@res_x
            res = res_x.flatten()
            return res
        return CP_mult_aux
    
    def Implicit_function(self):
        def aux(x):
            var = np.concatenate((x[0:self.n], self.x.flatten(), self.u.flatten(), x[self.n:])) 
            var_alt = np.concatenate((x[0:self.n], self.x.flatten(), self.u.flatten())) 
            return self.CP_multiplication()(var)
        return aux

    def Implicit_diff(self):
        DF = tmethods.diffenrentiation_CP(self.H_factors)
        def res(x):
            res = []
            x_total = np.concatenate((x[0:self.n], self.x.flatten(), self.u.flatten(), x[self.n:])) 
            for (i, D_Fi) in enumerate(DF):
                res.append(tmethods.CP_MTI_product(D_Fi, x_total))
            J = np.array(res).T
            len_xdot = len(x[0:self.n])
            len_y = len(x[self.n:])
            res = np.zeros((J.shape[0], len(x) ))
            print(f"shape {res.shape}")
            res[:,:len_xdot] = J[:,:len_xdot]
            if len_y != 0:
                res[:,len_xdot:] = J[:,-len_y:]
            return res
        return res 

    def series_MTIconnection(self, imti2, CP_factorised = False):
        #series conncection of a new iMTI after the existing iMTI
        #u2 = y1 the new equation
        H1 = self.H
        H2 = imti2.H

        m_new = self.m 
        p_new = imti2.p
        n_new = self.n + imti2.n
        e_new = self.e + imti2.e + self.p

        vars_h1 = 2*self.n + self.p + self.m
        vars_h2 = 2*self.n + self.p + self.m
        H_new = tuple(2 for _ in range(vars_h1 + vars_h2))
        H_shape = H_new + (e_new,)
        
        H1_relative = tuple(slice(0,2) for _ in range(vars_h1)) + tuple(0 for _ in range(vars_h2)) + (slice(0,self.e),)
        H2_relative = tuple(0 for _ in range(vars_h1)) + tuple(slice(0,2) for _ in range(vars_h2)) + (slice(self.e, 2*self.e),)
        H_series = tl.zeros(H_shape) 
        H_series[H1_relative] = H1
        H_series[H2_relative] = H2

        #We now add the constraint to the system
        H_equation_dimention = tuple(2 for _ in range(self.p)) + (self.p,)
        H_y1 = tuple(0 for _ in range(2*self.n + self.m)) + tuple(slice(0,2) for _ in range(self.p)) + tuple(0 for _ in range(vars_h2))  + (slice(2*self.e, e_new),)
        H_u2 = tuple(0 for _ in range(vars_h1 + 2*imti2.n)) + tuple(slice(0,2) for _ in range(imti2.m)) + tuple(0 for _ in range(imti2.p)) + (slice(2*self.e, e_new),)

        H_series[H_y1] = tl.ones(*H_equation_dimention) 
        H_series[H_u2] = -tl.ones(*H_equation_dimention)

        #Finally we move the axis accordingly to recover the structure H M(x', x, u, y)
        #In this case x = (x1, x2, y1, u2), y =(y2), u =(u1)
        #So from H  M(x'1, x1, u1, y1, x'2, x2, u2, y2) to H(x'1, x'2', x1, x2, y1, u2 , u1, y2)
        index_0 = list(range(0, self.n))
        index_1 = list(range(self.n, 2*self.n))
        index_2 = list(range(2*self.n, 2*self.n + self.m))
        index_3 = list(range(2*self.n + self.m, vars_h1))
        index_4 = list(range(vars_h1, vars_h1 + imti2.n))
        index_5 = list(range(vars_h1+ imti2.n, vars_h1 + 2*imti2.n))
        index_6 = list(range(vars_h1+ 2*imti2.n, vars_h1 + 2*imti2.n + imti2.m))
        source_index = [index_1, index_2, index_3, index_4, index_5, index_6]
        target_index = [index_2, index_6, index_4, index_1, index_3, index_5]
        H_series = tl.moveaxis(H_series, source_index, target_index)

        n_new = self.n + imti2.n + self.m + imti2.p
        # H(x'1, x'2', x1, x2, y1, u2 , u1, y2) to   H(x'1, x'2', y1', u2', x1, x2, y1, u2 , u1, y2)
        H_newshape = H_series.shape[0:self.n +imti2.n] + (2,)*(self.p + imti2.m) + H_series.shape[self.n +imti2.n:]
        H_broadcast = H_series.shape[0:self.n +imti2.n] + (0,)*(self.p + imti2.m) + H_series.shape[self.n +imti2.n:]
        H_final = tl.zeros(H_newshape)
        H_final[H_broadcast] = H_series
        
        series_mti = iMTI(n_new, m_new, p_new, e_new, H_final)
        return series_mti
    
    def parallel_MTIconnection(self, imti2):
        H1 = self.H
        H2 = imti2.H

        m_new = self.m 
        p_new = imti2.p
        e_new = self.e + imti2.e + self.p + self.m

        vars_h1 = 2*self.n + self.p + self.m
        vars_h2 = 2*self.n + self.p + self.m
        H_new = tuple(2 for _ in range(vars_h1 + vars_h2 + self.p))
        H_shape = H_new + (e_new,)

        H1_relative = tuple(slice(0,2) for _ in range(vars_h1)) + tuple(0 for _ in range(vars_h2)) + (slice(0,self.e),)
        H2_relative = tuple(0 for _ in range(vars_h1)) + tuple(slice(0,2) for _ in range(vars_h2)) + (slice(self.e, 2*self.e),)
        H_series = tl.zeros(H_shape) 
        H_series[H1_relative] = H1
        H_series[H2_relative] = H2

        #We now add the constraint to the system
        H_y = tuple(0 for _ in range(vars_h1 +vars_h2)) + tuple(slice(0,2) for _ in range(self.p)) + (slice(2*self.e, 2*self.e + self.p),)
        H_y1 = tuple(0 for _ in range(2*self.n + self.m)) + tuple(slice(0,2) for _ in range(self.p)) + tuple(0 for _ in range(vars_h2 + self.p))  + (slice(2*self.e, 2*self.e + self.p),)
        H_y2 = tuple(0 for _ in range(vars_h1 + 2*imti2.n + imti2.m)) + tuple(slice(0,2) for _ in range(imti2.p)) + tuple(0 for _ in range(self.p)) + (slice(2*self.e, 2*self.e + self.p),)
        H_series[H_y] = tl.ones(*H_y.shape)
        H_series[H_y1] = -tl.ones(*H_y1.shape)
        H_series[H_y2] = -tl.ones(*H_y2.shape)
        
        H_u1 = tuple(0 for _ in range(2*self.n)) + tuple(slice(0,2) for _ in range(self.m)) + tuple(0 for _ in range(self.p+vars_h2+self.p))+ (slice(2*self.e + self.p, None),)
        H_u2 = tuple(0 for _ in range(vars_h1 + 2*imti2.n)) + tuple(slice(0,2) for _ in range(imti2.m)) + tuple(0 for _ in range(imti2.p + self.p)) + (slice(2*self.e + self.p, None),)
        H_series[H_u1] = tl.ones(*H_u1.shape)
        H_series[H_u2] = -tl.ones(*H_u2.shape)

        #Then we move the axis accordingly to recover the structure H M(x', x, u, y)
        #So from H M(x'1, x1, u1, y1, x'2, x2, u2, y2, y) to H(x'1, x'2', x1, x2, y1, y2, u1, u2, y) 
        #the intermediate outputs become states
        index_1 = list(range(self.n, 2*self.n))
        index_2 = list(range(2*self.n, 2*self.n + self.m))
        index_3 = list(range(2*self.n + self.m, vars_h1))
        index_4 = list(range(vars_h1, vars_h1 + imti2.n))
        index_5 = list(range(vars_h1+ imti2.n, vars_h1 + 2*imti2.n))
        index_6 = list(range(vars_h1+ 2*imti2.n, vars_h1 + 2*imti2.n + imti2.m))
        index_7 = list(range(vars_h1+ 2*imti2.n + imti2.m, vars_h1 + vars_h2))
        source_index = index_1 + index_2 + index_3 + index_4 + index_5 + index_6 + index_7
        target_index = index_2 + index_7 + index_4 + index_1 + index_3 + index_7 + index_5
        H_series = tl.moveaxis(H_series, source_index, target_index)

        n_new = self.n + imti2.n + self.p + self.m + imti2.p
        # H(x'1, x'2', x1, x2, y1, y2, u1, u2, y) to  H(x'1, x'2',  y1', y2', u1', u2', x1, x2, y1, y2, u1, u2, y)
        #In this case x = (x1, x2, y1, y2, u1), y =(y), u =(u2)
        
        H_newshape = H_series.shape[0:self.n +imti2.n] + (2,)*(self.p + self.m + imti2.p) + H_series.shape[self.n +imti2.n:]
        H_broadcast = H_series.shape[0:self.n +imti2.n] + (0,)*(self.p + self.m + imti2.p) + H_series.shape[self.n +imti2.n:]
        H_final = tl.zeros(H_newshape)
        H_final[H_broadcast] = H_series
        
        parallel_mti = iMTI(n_new, m_new, p_new, e_new, H_final)
        return parallel_mti
    
    def feedback_MTIconnection(self, imti2):
        H1 = self.H
        H2 = imti2.H

        m_new = self.m 
        p_new = self.p
        n_new = self.n + imti2.n + self.m + imti2.m + imti2.p 
        e_new = self.e + imti2.e + self.m + self.p

        vars_h1 = 2*self.n + self.p + self.m
        vars_h2 = 2*self.n + self.p + self.m
        H_new = tuple(2 for _ in range(vars_h1 + vars_h2 + self.p))
        H_shape = H_new + (e_new,)

        H1_relative = tuple(slice(0,2) for _ in range(vars_h1)) + tuple(0 for _ in range(vars_h2)) + (slice(0,self.e),)
        H2_relative = tuple(0 for _ in range(vars_h1)) + tuple(slice(0,2) for _ in range(vars_h2)) + (slice(self.e, 2*self.e),)
        H_series = tl.zeros(H_shape) 
        H_series[H1_relative] = H1
        H_series[H2_relative] = H2

        #We now add the constraint to the system
        H_e = tuple(0 for _ in range(vars_h1 +vars_h2)) + tuple(slice(0,2) for _ in range(self.p)) + (slice(2*self.e, 2*self.e + self.p),)
        H_y2 = tuple(0 for _ in range(vars_h1 + 2*imti2.n + imti2.m)) + tuple(slice(0,2) for _ in range(imti2.p)) + tuple(0 for _ in range(self.p)) + (slice(2*self.e, 2*self.e + self.m),)
        H_u1 = tuple(0 for _ in range(2*self.n)) + tuple(slice(0,2) for _ in range(self.m)) + tuple(0 for _ in range(self.p+vars_h2+self.p))+ (slice(2*self.e, 2*self.e + self.m),)
        H_series[H_e] = tl.ones(*H_e.shape)
        H_series[H_u1] = -tl.ones(*H_u1.shape)
        H_series[H_y2] = tl.ones(*H_y2.shape)
        
        H_u2 = tuple(0 for _ in range(vars_h1 + 2*imti2.n)) + tuple(slice(0,2) for _ in range(imti2.m)) + tuple(0 for _ in range(imti2.p + self.p)) + (slice(2*self.e + self.m, None),)
        H_y1 = tuple(0 for _ in range(2*self.n + self.m)) + tuple(slice(0,2) for _ in range(self.p)) + tuple(0 for _ in range(vars_h2 + self.p))  + (slice(2*self.e + self.m, None),)
        H_series[H_y1] = tl.ones(*H_y1.shape)
        H_series[H_u2] = -tl.ones(*H_u2.shape)

        #Finally we move the axis accordingly to recover the structure H M(x', x, u, y)
        #So from H M(x'1, x1, u1, y1, x'2, x2, u2, y2, e) to H(x'1, x'2', x1, x2, y2, u1, u2, e, y1)) 
        #the intermediate outputs become states
        index_0 = list(range(self.n))
        index_1 = list(range(self.n, 2*self.n))
        index_2 = list(range(2*self.n, 2*self.n + self.m))
        index_3 = list(range(2*self.n + self.m, vars_h1))
        index_4 = list(range(vars_h1, vars_h1 + imti2.n))
        index_5 = list(range(vars_h1+ imti2.n, vars_h1 + 2*imti2.n))
        index_6 = list(range(vars_h1+ 2*imti2.n, vars_h1 + 2*imti2.n + imti2.m))
        index_7 = list(range(vars_h1+ 2*imti2.n + imti2.m, vars_h1 + vars_h2))
        index_8 = list(range(vars_h1 + vars_h2, vars_h1 + vars_h2 + self.m))
        source_index = index_1 + index_2 + index_3 + index_4 + index_5 + index_6 + index_7 + index_8
        target_index = index_2 + index_5 + index_8 + index_1 + index_3 + index_6 + index_4 + index_7
        H = tl.moveaxis(H, source_index, target_index)

        n_new = self.n + imti2.n + self.p + self.m + imti2.p+ imti2.m
        
        # H(x'1, x'2', x1, x2, y2, u1, u2, e, y1))  to  H(x'1, x'2',  y1', y2', u1', u2', x1, x2, y1, y2, u1, u2, e,y)
        #In this case x = (x1, x2, y2, u1, u2), y =(y1), u =(e)
        H_newshape = H_series.shape[0:self.n +imti2.n] + (2,)*(self.p + self.m + imti2.m) + H_series.shape[self.n +imti2.n:]
        H_broadcast = H_series.shape[0:self.n +imti2.n] + (0,)*(self.p + self.m + imti2.m) + H_series.shape[self.n +imti2.n:]
        H_final = tl.zeros(H_newshape)
        H_final[H_broadcast] = H_series

        mti_feedback = iMTI(n_new, m_new, p_new, e_new, H_final)
        return mti_feedback


class sMTI:
    def __init__(self, eMTI, algebra):
        self.eMTI = eMTI
        self.algebra = algebra

    def solve(self):
        return 0