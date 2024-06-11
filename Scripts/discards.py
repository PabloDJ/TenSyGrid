#DISCARDED FILES AND FUCNTIONS
#COULD BE USEFUL LATER

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


            for _ in range(5):
                for key in service_eqs.keys():
                    key_var = sp.Symbol(key)
                    eq = service_eqs[key]
                    if type(eq)==float or type(eq)==sp.core.numbers.Float:
                        for key2 in service_eqs.keys():
                            eq2 = service_eqs[key2]
                            if key_var in eq2.free_symbols:
                                service_eqs[key2] = eq2.subs(key_var, float(eq))
                            

def sys_to_tensor(sys, order =2):
    tensor_shape = (0,)*order + (0,)
    F = np.empty(tensor_shape)
    G = np.empty(tensor_shape)
    for mdl in sys.models.values():   
        num_devices = mdl.n
        try:
            mdl.prepare()
        except:
            print(mdl)
        try:
            vars_xy = mdl.syms.vars_list
            f_equations = mdl.syms.f_list
            g_equations = mdl.syms.g_list
            service_dict = mdl.syms.s_syms
            config_dict = mdl.config
            _ = 0
        except:
            continue
        
        for device in range(num_devices):
            if type(mdl).__name__ == "Toggle":
                continue
            elif type(mdl).__name__ == "Bus":
                bus_id = mdl.idx.v[device]
                bus_ids = [bus_id]
            elif type(mdl).__name__ == "Line":
                bus_id_from = mdl.bus1.v[device]
                bus_id_to = mdl.bus2.v[device]
                bus_ids = [bus_id_from , bus_id_to]
                
            elif mdl.group in ["TurbineGov", "Exciter"]:
                generator = "GENROU"
                gen = getattr(sys, generator)
                syn_id = int(mdl.syn.v[device])
                bus_id = gen.bus.v[syn_id-1]
                bus_ids = [bus_id]
                
            else:
                try:
                    print(mdl.bus)
                    bus_id = mdl.bus.v[device] 
                    bus_ids = [bus_id]
                except:
                    print(f"model {mdl} does not contribute equations")
            
            print(f"model out {mdl}")
            bus_vars = set()
            if len(bus_ids) == 1:
                bus_vars = {sp.Symbol("a"), sp.Symbol("v")}
            else: 
                for i, bus in enumerate(bus_ids):
                    bus_vars = bus_vars|{sp.Symbol("a"+str(i+1)), sp.Symbol("v"+str(i+1))}
            
            vars = set(mdl.algebs.keys())|set(mdl.algebs_ext.keys())|set(mdl.states.keys())|set(mdl.states_ext.keys())
            vars = vars|bus_vars
            vars = {sp.Symbol(s) if isinstance(s, str) else s for s in vars}
            
            vars = list(vars)
                
            f_equations = [eq for eq in f_equations if not (isinstance(eq, float) or isinstance(eq, int))]
            g_equations = [eq for eq in g_equations if not (isinstance(eq, float) or isinstance(eq, int))]

            initial_vars = service_dict.keys()
            final_vars = service_dict.values()
            for i in range(3):
                f_equations = var_substitution(f_equations, initial_vars, final_vars)
                g_equations = var_substitution(g_equations, initial_vars, final_vars)  
            
            if len(f_equations)!= 0:
                F_res_sympy, F_res_float = symbolic_to_tensor(vars , f_equations)
                F_res_substitue = param_substitution(F_res_sympy, mdl, device)
                F = update_tensor(F_res_substitue, F, bus_id) 
            
            if len(g_equations)!= 0:
                G_res_sympy, G_res_float = symbolic_to_tensor(vars , g_equations)
                G_res_substitue = param_substitution(G_res_sympy, mdl, device)
                G = update_tensor(G_res_substitue, G, bus_id)
                        
    return F, G
            