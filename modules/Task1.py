import numpy as np
from modules.Task0 import apply_diffusion_advection

def arnoldi_iteration(A, V, H, r_norm, m):
    """
    Perform Arnoldi iteration to construct the Krylov subspace.
    
    Parameters:
    - A: Function to apply the operator (matrix-free implementation).
    - V: Orthonormal basis of the Krylov subspace (array of size [len(r), m+1]).
    - H: Hessenberg matrix (array of size [m+1, m]).
    - r_norm: Norm of the initial residual.
    - m: Subspace dimension.

    Returns:
    - V: Updated Krylov subspace basis.
    - H: Updated Hessenberg matrix.
    """
    for j in range(m):
        #Apply the operator
        w = A(V[:, j])
        
        #Orthogonalize
        for i in range(j+1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]
        
        #Normalize and store the next basis vector
        H[j+1, j] = np.linalg.norm(w)
        if H[j+1, j] > 0 and j+1 < m:
            V[:, j+1] = w / H[j+1, j]
    
    return V, H

def solve_least_squares(H, r_norm, m):
    """
    Solve the least squares problem in the Krylov subspace.
    
    Parameters:
    - H: Hessenberg matrix (array of size [m+1, m]).
    - r_norm: Norm of the initial residual.
    - m: Subspace dimension.

    Returns:
    - y: Coefficients for the approximate solution in the Krylov subspace.
    """
    #Construct the right-hand side for the least squares problem
    e1 = np.zeros(m+1)
    e1[0] = r_norm  # First element is the norm of the residual
    
    #Solve the least squares problem H @ y ≈ e1
    y, _, _, _ = np.linalg.lstsq(H[:m+1, :m], e1, rcond=None)
    return y

import numpy as np
import time

def restarted_gmres_grid(U, f, N, h, v, tol, m, max_restarts=10, verbose=-1):
    """
    Restarted GMRES directly on the grid, using matvec_with_boundary.
    Updated to handle zero residuals explicitly and prevent breakdowns.
    
    Parameters:
        U: Initial guess for the solution (N+1)x(N+1) array.
        f: Right-hand side (N+1)x(N+1) array.
        N: Number of grid intervals.
        h: Grid spacing.
        v: Velocity field (v1, v2).
        tol: Convergence tolerance.
        m: Number of iterations before restarting.
        max_restarts: Maximum number of GMRES restarts.
        verbose: Verbosity level (-1: no output, 0: summary, 1: detailed).

    Returns:
        U: Approximate solution (N+1)x(N+1) array.
        residuals: Residuals at each restart.
        elapsed_time: Time taken to converge (or reach max restarts).
    """
    def matvec_with_boundary(V_flat):
        """
        Apply the operator with explicit boundary condition enforcement.
        """
        V = V_flat.reshape((N+1, N+1))
        AV = apply_diffusion_advection(V, h, N, v)
        AV[0, :] = 0
        AV[-1, :] = 0
        AV[:, 0] = 0
        AV[:, -1] = 0
        return AV.flatten()

    #Flatten U and f
    U_flat = U.flatten()
    f_flat = f.flatten()

    #Compute initial residual
    r = f_flat - matvec_with_boundary(U_flat)
    r_grid = r.reshape((N+1, N+1))
    r_grid[0, :] = 0
    r_grid[-1, :] = 0
    r_grid[:, 0] = 0
    r_grid[:, -1] = 0
    r = r_grid.flatten()
    r_norm = np.linalg.norm(r)
    r0_norm = r_norm  #Initial residual norm
    residuals = [r_norm]

    if r_norm == 0 or r_norm / r0_norm < tol:
        if verbose >= 0:
            print("Initial residual norm is zero or already within tolerance.")
        return U, residuals, 0.0

    start_time = time.perf_counter()

    for restart in range(max_restarts):
        V = np.zeros((len(r), m+1))
        H = np.zeros((m+1, m))
        
        V[:, 0] = r / r_norm if r_norm > 0 else 0

        #Arnoldi iteration
        V, H = arnoldi_iteration(matvec_with_boundary, V, H, r_norm, m)

        #Solve the least squares problem
        try:
            y = solve_least_squares(H, r_norm, m)
        except np.linalg.LinAlgError as e:
            print(f"Least squares solver failed: {e}")
            elapsed_time = time.perf_counter() - start_time
            return U, residuals, elapsed_time

        #Update the solution
        U_flat = U_flat + V[:, :m] @ y

        #Compute new residual
        r = f_flat - matvec_with_boundary(U_flat)
        r_grid = r.reshape((N+1, N+1))
        
        #Enforce the boundary
        r_grid[0, :] = 0
        r_grid[-1, :] = 0
        r_grid[:, 0] = 0
        r_grid[:, -1] = 0
        r = r_grid.flatten()
        r_norm = np.linalg.norm(r)
        residuals.append(r_norm)

        if r_norm / r0_norm < tol:
            elapsed_time = time.perf_counter() - start_time
            if verbose >= 0:
                print(f"Converged after {restart+1} restarts with residual: {r_norm / r0_norm:.2e}")
            return U_flat.reshape((N+1, N+1)), residuals, elapsed_time
        if verbose == 1:
            print(f"Restart {restart+1}/{max_restarts}, residual: {r_norm / r0_norm:.2e}")

    elapsed_time = time.perf_counter() - start_time
    print("Failed to converge within maximum restarts.")
    return U_flat.reshape((N+1, N+1)), residuals, elapsed_time



