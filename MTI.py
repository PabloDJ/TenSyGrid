import numpy as np
import tensorly as tl

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
    def __init__(self, n, m, p, H):
        self.n = n
        self.m = m
        self.p = p
        self.H = H

    def CP_decomposition(self, rank = 5):
        H_tensor = tl.tensor(self.H, rank)
        self.F_factors = tl.decomposition.parafac(H_tensor,rank)

class sMTI:
    def __init__(self, eMTI, algebra):
        self.eMTI = eMTI
        self.algebra = algebra

    def solve(self):
        return 0