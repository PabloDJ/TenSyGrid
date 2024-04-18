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
    y = np.zeros_like(t)
    y[0] = y0
    
    # Time step
    dt = t[1] - t[0]
    
    # Perform trapezoidal rule iteration
    for i in range(1, len(t)):
        # Define the function to be solved implicitly
        def residual(y_next):
            return y_next - y[i-1] - 0.5 * dt * (func(t[i], y_next) + func(t[i-1], y[i-1]))
        
        # Solve the nonlinear equation using fsolve
        y[i] = fsolve(residual, y[i-1])[0]
        
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
        y_next, _, info, msg = fsolve(F_next, y_values[i], fprime=J_next, full_output=True)
        if info != 1:
            raise RuntimeError(f"Nonlinear solver did not converge: {msg}")
        y_values[i + 1] = y_next

    return y_values