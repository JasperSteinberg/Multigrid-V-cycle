# modules/functions.py
import numpy as np

def u_exact(x,y):
    """
    Calculate the exact solution.

    Args:
        x (float)
        y (float)

    Returns:
        float: The calculated exact solution.
    """ 
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)

def rhs_func(x,y):
    
    """
    Calculate the exact right hand side.

    Args:
        x (float)
        y (float)

    Returns:
        float: The calculated right hand side.
    """
    
    return (5*np.pi**2)*u_exact(x,y) + np.pi*np.cos(np.pi*x)*np.sin(2*np.pi*y) + 2*np.pi*np.sin(np.pi*x)*np.cos(2*np.pi*y)

# Define the grid and exact solution
def setup_test(N):
    h = 1 / N
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    
    # Exact solution
    u = u_exact(X,Y)
    
    # Compute f(x, y) for the given exact solution
    f = rhs_func(X,Y)
    
    return X, Y, u, f, h
