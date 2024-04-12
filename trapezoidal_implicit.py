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
    
