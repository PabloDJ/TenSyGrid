import numpy as np
import tensorly as tl

class eMTI:
    def __init__(self, n, m, p, F, G):
        self.n = n
        self.m = m
        self.p = p
        self.F = F
        self.G = G
    
    def CP_decomposition(self, rank = 6):
        F_tensor = tl.tensor(self.F)
        G_tensor = tl.tensor(self.G)
        self.F_factors = tl.decomposition.parafac(F_tensor,rank)
        self.G_factors = tl.decomposition.parafac(G_tensor,rank)
    
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

    def CP_decomposition(self, rank = 6):
        H_tensor = tl.tensor(self.H, rank)
        self.F_factors = tl.decomposition.parafac(H_tensor,rank)