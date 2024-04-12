def func(F_CP):
    def res(x_save):
        x = x_save[0:n]
        u = x_save[n:n+m]
        aux = np.zeros(n+m)
        aux[0:n] = methods.MTI_product(F_CP, x, u)
        return aux
    return res

f = func(eMTI.F_factors)

def jacob(F_CP):
    DF = methods.diffenrentiation_CP(0, F_CP)
    def res(x_save):
        aux = np.zeros((n+m,n+m))
        x = x_save[0:n]
        u = x_save[n:n+m]
        for (i, D_Fi) in enumerate(DF):
            aux[i,0:2] = methods.MTI_product(D_Fi, x, u)
        return aux
    return res 

j = jacob(eMTI.F_factors)