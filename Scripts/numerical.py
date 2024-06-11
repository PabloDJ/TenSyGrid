import numpy as np
import tensorly as tl
from scipy.optimize import fsolve, root
from scipy.linalg import solve
import plot

def newton_raphson_multivariate(f, x0, tol=1e-6, max_iter=100):
    """Newton-Raphson method for solving a system of equations with n variables and m outputs."""
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        J = jacobian(f, x)
        F_val = np.array(f(x))
        # Solve for delta x using the linear system J(x) * delta_x = -F(x)
        try:
            delta_x = solve(J, -F_val)
        except np.linalg.LinAlgError:
            print("Jacobian is singular, cannot solve.")
            return None
        # Update the estimate
        x += delta_x
        # Check for convergence
        if np.linalg.norm(delta_x) < tol:
            return x
    print("Did not converge within the maximum number of iterations.")
    return None

def trapezoidal_implicit(func, y0, t):
    """
    Trapezoidal method with implicit method for solving ODEs.
    
    Parameters:
        func : function
            The function defining the ODE dy/dt = func(t, y).
        y0 : float or array_like
            Initial condition(s) of the dependent variable.
        t : array_like
            Array of time points at which to solve for y.
            
    Returns:
        y : array_like
            Array containing the solution at each time point.
    """
    # Initialize solution array
    y = np.zeros((len(t), len(y0))) 
    y[0,:] = y0
    
    # Time step
    dt = t[1] - t[0]
    
    # Perform trapezoidal rule iteration
    for i in range(1, len(t)):
        # Define the function to be solved implicitly
        def residual(y_next):
            return y_next - y[i-1,:] - 0.5 * dt * (func( y_next, t[i]) + func(y_next, t[i-1]))
        
        # Solve the nonlinear equation using fsolve
        y[i,:] = fsolve(residual, y[i-1,:])[0]
        
    return y
    
def backward_euler_DAE(F, J, y0, t):
    """
    Solves the DAE system of the form F(y, dy/dt, t) = 0 using the Backward Euler method.

    Parameters:
        F : function
            The function F(y, dy/dt, t) defining the DAE.
        J : function
            The Jacobian of F with respect to y.
        y0 : numpy.ndarray
            Initial state vector.
        t : numpy.ndarray
            Timesteps vector
        
    Returns:
        t_values : numpy.ndarray
            Array of time values.
        y_values : numpy.ndarray
            Matrix of state vectors at each time step.
    """
    
    y_values = np.zeros((len(t) + 1, len(y0)))
    y_values[0] = y0

    for i in range(1, len(t)):
        h = t[i]-t[i-1]
        F_next = lambda y: F(y, (y - y_values[i]) / h, t)
        J_next = lambda y: J(y, (y - y_values[i]) / h, t)
        y_next, info, ier, msg = fsolve(F_next, y_values[i], fprime=J_next, full_output=True)
        if ier != 1:
            xz_next =  y_values[i]
            print(f" f(x) is {F_next(y_next)}")
            raise RuntimeError(f"Nonlinear solver did not converge: {msg}")
        y_values[i + 1] = y_next

    return y_values

def backward_euler_semi_explicit(f, g, x0, z0, t, DAE_Jacob = False, binary = True):
    """
    Solves the semi-explicit DAE system where dot(x) = f(x, z, t) and 0 = g(x, z, t).
    
    Parameters:
        f : function
            Function that computes the derivative dot(x).
        g : function
            Algebraic constraint function g(x, z, t).
        x0 : numpy.ndarray
            Initial differential state vector.
        z0 : numpy.ndarray
            Initial algebraic state vector.
        t0 : float
            Initial time.
        tf : float
            Final time.
        h : float
            Time step size.
    Returns:
        t_values : numpy.ndarray
            Array of time values.
        x_values : numpy.ndarray
            Matrix of differential states at each time step.
        z_values : numpy.ndarray
            Matrix of algebraic states at each time step.
    """
    num_steps = len(t)-1
    h = t[1] - t[0]
    print(f"x0 is of length {len(x0)}")
    print(f"z0 is of length {len(z0)}")
    x_values = np.zeros((num_steps + 1, len(x0)))
    z_values = np.zeros((num_steps + 1, len(z0)))
    t_values = t
    x_values[0], z_values[0] = x0, z0

    for i in range(num_steps):
        t_next = t_values[i + 1]
        def solver_func(xz):
            if not binary:
                res = np.concatenate([
                xz[:len(x0)] - x_values[i] - h * f(xz[:len(x0)], xz[len(x0):], t_next),
                g(xz[:len(x0)], xz[len(x0):], t_next)
                ])
            else:
                n_aux = len(f(xz[:len(x0)], xz[len(x0):], t_next))
                res = np.concatenate([
                xz[:n_aux] - x_values[i,:n_aux] - h * f(xz[:len(x0)], xz[len(x0):], t_next),
                g(xz[:len(x0)], xz[len(x0):], t_next)
                ])

            return res
        if type(DAE_Jacob) == bool:
            sol  = root(solver_func, np.concatenate([x_values[i], z_values[i]]))
            xz_next = sol.x
            ier = sol.success
            msg = sol.message
        else:
            def DAE_J(xz):
                res = DAE_Jacob(xz[:len(x0)], xz[len(x0):], t)
                res[:len(x0),:] = - h*res[:len(x0), :] 
                res[:len(x0),:len(x0)] += np.eye(len(x0))
                return res
            sol  = root(solver_func, np.concatenate([x_values[i], z_values[i]]))
            xz_next = sol.x
            ier = sol.success
            msg = sol.message
            sol = newton_raphson_multivariate(solver_func, np.concatenate([x_values[i], z_values[i]]))
        if ier != 1:
            v = np.concatenate([x_values[i], z_values[i]])
            print(f" f(x) is {solver_func(xz_next)}")
            print("time is", t_next)
            print("message ", msg)
            J = jacobian(solver_func, xz_next)
            print("determinant is ", np.linalg.det(J))
            #sol = newton_raphson_multivariate(solver_func, np.concatenate([x_values[i], z_values[i]]))
            print("N_R gives the solution ", xz_next)
            plot.vicinity(solver_func, xz_next, verbose = True)
            raise RuntimeError("Nonlinear solver did not converge")
        x_values[i + 1], z_values[i + 1] = xz_next[:len(x0)], xz_next[len(x0):]
    return  x_values, z_values

def algebraic_substitution_semi_explicit(f, g, x0, z0, t, DAE_Jacob = False, z_explicit = False):
    """
    Solves the semi-explicit DAE system where dot(x) = f(x, z, t) and 0 = g(x, z, t).
    
    Parameters:
        f : function
            Function that computes the derivative dot(x).
        g : function
            Algebraic constraint function g(x, z, t).
        x0 : numpy.ndarray
            Initial differential state vector.
        z0 : numpy.ndarray
            Initial algebraic state vector.
        t0 : float
            Initial time.
        tf : float
            Final time.
        h : float
            Time step size.
    Returns:
        t_values : numpy.ndarray
            Array of time values.
        x_values : numpy.ndarray
            Matrix of differential states at each time step.
        z_values : numpy.ndarray
            Matrix of algebraic states at each time step.
    """
    num_steps = len(t)-1
    h = t[1] - t[0]
    x_values = np.zeros((num_steps + 1, len(x0)))
    z_values = np.zeros((num_steps + 1, len(z0)))
    t_values = t
    x_values[0], z_values[0] = x0, z0

    for i in range(num_steps):
        t_next = t_values[i + 1]
        if type(z_explicit) == bool: 
            def z_implicit(x, t):
                def g_z(z): 
                    return g(x, z, t)
                sol = root(g_z, z0)
                z_implicit = sol.x
                return z_implicit
        else: 
            z_implicit = z_explicit
        def solver_func(x):
            z_impl = z_implicit(x, t_next)
            res = x - x_values[i] - h * f(x, z_impl, t_next)
            return res
        if type(DAE_Jacob) == bool:
            sol  = root(solver_func, x_values[i])
            x_next = sol.x
            ier = sol.success
            msg = sol.message
        else:
            def DAE_J(xz):
                res = DAE_Jacob(xz[:len(x0)], xz[len(x0):], t)
                res[:len(x0),:] = - h*res[:len(x0), :] 
                res[:len(x0),:len(x0)] += np.eye(len(x0))
                return res
            sol  = root(solver_func, x_values[i], jac=DAE_J)
            x_next = sol.x
            ier = sol.success
            msg = sol.message
        if ier != 1:
            print(ier)
            print(msg)
            print("time is", t_next)
            xz_next = np.concatenate([x_values[i], z_values[i]])
            print(f" Jacobian is {jacobian(solver_func, xz_next)}")
            plot.vicinity(solver_func, x0, verbose = True)
            raise RuntimeError("Nonlinear solver did not converge")
        x_values[i + 1] = x_next
        z_values[i + 1] = z_implicit(x_next, t_next)

    return  x_values, z_values

def jacobian(f, x0):
    epsilon = 1e-6

    # Initialize an empty Jacobian matrix
    Jacobian = np.zeros((len(f(x0)), len(x0)))

    # Compute the Jacobian matrix numerically using finite differences
    for i in range(len(x0)):
        x_plus_epsilon = np.array(x0)
        x_plus_epsilon[i] += epsilon
        J_plus_epsilon = f(x_plus_epsilon)

        x_minus_epsilon = np.array(x0)
        x_minus_epsilon[i] -= epsilon
        J_minus_epsilon = f(x_minus_epsilon)

        Jacobian[:, i] = (J_plus_epsilon - J_minus_epsilon) / (2 * epsilon)
        
    return Jacobian

def numerical_hessian(func, x0, epsilon=1e-5):
    """
    Compute the numerical Hessian matrix of a scalar-valued function.
    
    Parameters:
    - func: A scalar-valued function of a vector input
    - x0: Initial point (numpy array) at which to compute the Hessian
    - epsilon: Step size for finite difference
    
    Returns:
    - Hessian matrix (numpy array of shape (n, n))
    """
    n = len(x0)
    hessian = np.zeros((n, n))
    perturb = np.zeros(n)

    # Compute the Hessian using finite differences
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal elements (second derivatives)
                perturb[i] = epsilon
                f1 = func(x0 + perturb)
                f2 = func(x0 - perturb)
                f0 = func(x0)
                hessian[i, j] = (f1 - 2 * f0 + f2) / (epsilon**2)
                perturb[i] = 0
            else:
                # Off-diagonal elements (mixed partial derivatives)
                perturb[i] = epsilon
                perturb[j] = epsilon
                f1 = func(x0 + perturb)
                f2 = func(x0 - perturb)
                f3 = func(x0 + np.array([epsilon, -epsilon]) * np.array([i == 0, j == 0]))
                f4 = func(x0 + np.array([-epsilon, epsilon]) * np.array([i == 0, j == 0]))
                hessian[i, j] = (f1 - f2 - f3 + f4) / (4 * epsilon**2)
                perturb[i] = 0
                perturb[j] = 0
    return hessian