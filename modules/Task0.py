import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib.pyplot as plt

def apply_diffusion_advection(U, h, N, v):
    """
    Apply the discrete diffusion-advection operation using vectorized operations.
    
    Parameters:
    - U: (N+1)x(N+1) numpy array, matrix input to the operator
    - h: float, stepsize
    - N: int, grid size (number of intervals along one dimension)
    - v: tuple, velocity field (v1, v2)

    Returns:
    - result: The matrix after the diffusion-advection operator
    """
    v1, v2 = v  #Velocity components

    #Internal grid points
    index = np.arange(1, N)
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index-1, index)
    ixp_y = np.ix_(index+1, index)
    ix_ym = np.ix_(index, index-1)
    ix_yp = np.ix_(index, index+1)

    #Initialize result array
    result = np.zeros_like(U)

    #Apply operator to internal points only
    result[ixy] = (
        4 * U[ixy]
        - U[ixp_y] - U[ixm_y]
        - U[ix_yp] - U[ix_ym]
        + h * v1 * (U[ixy] - U[ixm_y])
        + h * v2 * (U[ixy] - U[ix_ym])
    ) / h**2

    #Ensure boundary points remain untouched
    result[0, :] = 0
    result[-1, :] = 0
    result[:, 0] = 0
    result[:, -1] = 0

    return result


def matvec_with_boundary(V_flat, N, h, v):
    """
    Apply the operator with explicit boundary condition enforcement.
    
    Parameters:
    - V_flat: Flattened (N+1)*(N+1) array representing the solution vector.
    - N: Number of grid intervals.
    - h: Grid spacing.
    - v: Velocity field as a tuple (v1, v2).
    
    Returns:
    - Flattened array after applying the operator.
    """
    #Reshape the flat vector into a grid
    V = V_flat.reshape((N+1, N+1))
    
    #Apply the operator
    AV = apply_diffusion_advection(V, h, N, v)
    
    #Enforce boundary conditions explicitly
    AV[0, :] = 0
    AV[-1, :] = 0
    AV[:, 0] = 0
    AV[:, -1] = 0
    
    #Flatten back to a vector
    return AV.flatten()

