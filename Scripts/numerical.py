import numpy as np
import tensorly as tl
from scipy.optimize import fsolve


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
            raise RuntimeError(f"Nonlinear solver did not converge: {msg}")
        y_values[i + 1] = y_next

    return y_values

def backward_euler_semi_explicit(f, g, x0, z0, t, DAE_Jacob = False):
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
        def solver_func(xz):
            res =np.concatenate([
            xz[:len(x0)] - x_values[i] - h * f(xz[:len(x0)], xz[len(x0):], t_next),
            g(xz[:len(x0)], xz[len(x0):], t_next)
            ])

            return res
        if type(DAE_Jacob) == bool:
            xz_next, info, ier, msg = fsolve(solver_func, np.concatenate([x_values[i], z_values[i]]), full_output=True)
        else:
            def DAE_J(xz):
                res = DAE_Jacob(xz[:len(x0)], xz[len(x0):], t)
                res[:len(x0),:] = - h*res[:len(x0), :] 
                res[:len(x0),:len(x0)] += np.eye(len(x0))
                return res
            xz_next, info, ier, msg = fsolve(solver_func, np.concatenate([x_values[i], z_values[i]]), fprime=DAE_J, full_output=True)
        if ier != 1:
            print(ier)
            print(msg)
            print("time is", t_next)
            raise RuntimeError("Nonlinear solver did not converge")
        x_values[i + 1], z_values[i + 1] = xz_next[:len(x0)], xz_next[len(x0):]

    return  x_values, z_values