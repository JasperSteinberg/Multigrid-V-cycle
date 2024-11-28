import numpy as np
from modules.functions import u_exact, rhs_func, setup_test
from modules.Task1 import restarted_gmres_grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def jacobi(u, rhs, omega, N, iterations, v=(1, 1)):
    """
    Perform Jacobi iterations for the advection-diffusion equation.

    Parameters:
    - u: (N+1)x(N+1) numpy array, initial guess for the solution
    - rhs: (N+1)x(N+1) numpy array, right-hand side
    - omega: float, relaxation parameter (e.g., 2/3)
    - N: int, grid size (number of intervals along one dimension)
    - iterations: int, number of Jacobi iterations to perform
    - v: tuple, velocity field (v1, v2)

    Returns:
    - u: Updated solution after `iterations` Jacobi steps
    """
    h = 1 / N  #Grid spacing
    v1, v2 = v

    #Prepare indices for the interior points
    index = np.arange(1, N)
    ixy = np.ix_(index, index)  #Interior points
    ixm_y = np.ix_(index - 1, index)  #(i-1, j)
    ixp_y = np.ix_(index + 1, index)  #(i+1, j)
    ix_ym = np.ix_(index, index - 1)  #(i, j-1)
    ix_yp = np.ix_(index, index + 1)  #(i, j+1)

    #Denominator including advection terms
    div = (4 + v1 * h + v2 * h)

    for _ in range(iterations):
        #Save the current solution to avoid overwriting
        u_old = u.copy()
        
        #Update the interior points using the Jacobi formula
        u[ixy] = (
            (u_old[ixp_y] + u_old[ixm_y] * (1 + v1 * h) +
             u_old[ix_yp] + u_old[ix_ym] * (1 + v2 * h)) +
             h**2*rhs[ixy]
        ) / div
        
        #Apply relaxation (weighted update)
        u[ixy] = omega * u[ixy] + (1 - omega) * u_old[ixy]

    return u


def test_jacobi(N = 64, iterations = 4000, v=[1,1]):
    
    X,Y,exact,rhs,h = setup_test(N)

    #Initial guess for the solution
    u_initial = np.zeros((N+1, N+1))


    #Solve using Jacobi iterations
    omega = 2/3
    u_computed = jacobi(u_initial, rhs, omega, N, iterations, v)
    print(f'Maximum error: {np.max(u_computed - exact)}')

    #Plot the exact solution and computed solution for comparison
    fig = plt.figure(figsize=(12, 6))

    #Exact solution
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, exact, cmap='viridis')
    ax1.set_title('Exact Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')

    #Computed solution
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, u_computed, cmap='viridis')
    ax2.set_title('Computed Solution (Jacobi)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')

    plt.tight_layout()
    plt.show()
    
def residual(u, rhs, N, v=(1, 1)):
    """
    Compute the residual r = f - (1/h^2)L(u) for the 2D advection-diffusion problem.

    Parameters:
    - u: (N+1)x(N+1) numpy array, current solution
    - rhs: (N+1)x(N+1) numpy array, right-hand side
    - N: int, grid size (number of intervals along one dimension)
    - v: tuple, velocity field (v1, v2)

    Returns:
    - r: (N+1)x(N+1) numpy array, residual (with boundary set to 0)
    """
    h = 1 / N  #Grid spacing
    v1, v2 = v

    #Prepare indices for the interior points
    index = np.arange(1, N)
    ixy = np.ix_(index, index)  #Interior points
    ixm_y = np.ix_(index - 1, index)  #(i-1, j)
    ixp_y = np.ix_(index + 1, index)  #(i+1, j)
    ix_ym = np.ix_(index, index - 1)  #(i, j-1)
    ix_yp = np.ix_(index, index + 1)  #(i, j+1)

    #Apply the operator L(u) to the interior points
    Lu = np.zeros_like(u)
    Lu[ixy] = (
        4 * u[ixy]
        - u[ixp_y] - u[ixm_y]
        - u[ix_yp] - u[ix_ym]
        + h * v1 * (u[ixy] - u[ixm_y])
        + h * v2 * (u[ixy] - u[ix_ym])
    )
    
    #Compute the residual for the interior points only
    r = np.zeros_like(u)
    r[ixy] = rhs[ixy] - Lu[ixy]/h**2

    return r

def test_residual(N, v=(1, 1)):
    """
    Plot the residual r = f - (1/h^2)L(u) for the exact solution of the 2D advection-diffusion problem.
    
    Parameters:
    - N: int, grid size (number of intervals along one dimension)
    - v: tuple, velocity field (v1, v2)
    """
    #Define grid and step size
    X,Y,exact,rhs,h = setup_test(N)

    #Compute the residual using the given function
    r = residual(exact, rhs, N, v)

    #Create the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, r, cmap='inferno', edgecolor='k')
    ax.set_title("3D Plot of Residual for Exact Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Residual")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()

    #Print maximum residual for verification
    max_residual = np.max(np.abs(r))
    print(f"Max residual for the exact solution: {max_residual:.2e}")


def restriction(fine, N_fine):
    """
    Restrict a fine grid (N_fine+1)x(N_fine+1) array to a coarse grid using full-weighting.

    Parameters:
    - fine: (N_fine+1)x(N_fine+1) numpy array, fine grid data
    - N_fine: int, number of intervals on the fine grid

    Returns:
    - coarse: (N_coarse+1)x(N_coarse+1) numpy array, coarse grid data
    """
    N_coarse = N_fine // 2  #Coarse grid has half the resolution
    coarse = np.zeros((N_coarse + 1, N_coarse + 1))

    #Indices for the interior points on the coarse grid
    i_coarse = np.arange(1, N_coarse)
    j_coarse = np.arange(1, N_coarse)
    i_fine = 2 * i_coarse
    j_fine = 2 * j_coarse

    #Apply full-weighting restriction for interior points
    coarse[np.ix_(i_coarse, j_coarse)] = (
        4 * fine[np.ix_(i_fine, j_fine)]  #Center points
        + 2 * (fine[np.ix_(i_fine + 1, j_fine)] + fine[np.ix_(i_fine - 1, j_fine)])  #Vertical neighbors
        + 2 * (fine[np.ix_(i_fine, j_fine + 1)] + fine[np.ix_(i_fine, j_fine - 1)])  #Horizontal neighbors
        + fine[np.ix_(i_fine + 1, j_fine + 1)]  #Diagonal neighbors
        + fine[np.ix_(i_fine + 1, j_fine - 1)]
        + fine[np.ix_(i_fine - 1, j_fine + 1)]
        + fine[np.ix_(i_fine - 1, j_fine - 1)]
    ) / 16

    #Boundary values are directly transferred
    coarse[0, :] = fine[0, ::2]  #Bottom boundary
    coarse[-1, :] = fine[-1, ::2]  #Top boundary
    coarse[:, 0] = fine[::2, 0]  #Left boundary
    coarse[:, -1] = fine[::2, -1]  #Right boundary

    return coarse

def test_restriction(N_fine):
    """
    Test the restriction function by comparing the exact solution on the coarse grid
    with the restricted solution from the fine grid, using 3D visualization.

    Parameters:
    - N_fine: Number of intervals on the fine grid.
    """
    #Fine grid setup
    X_fine,Y_fine,u_fine,_,h_fine = setup_test(N_fine)
    
    #Coarse grid setup
    N_coarse = N_fine // 2
    X_coarse,Y_coarse,u_coarse,_,h_coarse = setup_test(N_coarse)

    #Apply restriction
    u_coarse_restricted = restriction(u_fine, N_fine)

    #Visualize in 3D
    fig = plt.figure(figsize=(16, 6))

    #Exact coarse solution
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_coarse, Y_coarse, u_coarse, cmap="viridis")
    ax1.set_title("Exact Coarse Solution")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("u(x, y)")

    #Restricted coarse solution
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_coarse, Y_coarse, u_coarse_restricted, cmap="viridis")
    ax2.set_title("Restricted Coarse Solution")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("u(x, y)")

    plt.tight_layout()
    plt.show()

def interpolation(coarse, N_fine):
    """
    Interpolates from a coarse grid to a fine grid.
    coarse: coarse grid solution (shape (N_coarse+1, N_coarse+1))
    N_fine: fine grid size (excluding boundary points)
    Returns: fine grid solution (shape (N_fine+1, N_fine+1))
    """
    N_coarse = (N_fine - 1) // 2
    fine = np.zeros((N_fine + 1, N_fine + 1))
    
    #Copy coarse grid points directly
    fine[::2, ::2] = coarse  #Coarse points align with every second fine point
    
    #Interpolate horizontally
    fine[1::2, ::2] = 0.5 * (coarse[:-1, :] + coarse[1:, :])
    
    #Interpolate vertically
    fine[::2, 1::2] = 0.5 * (coarse[:, :-1] + coarse[:, 1:])
    
    #Interpolate diagonally
    fine[1::2, 1::2] = 0.25 * (coarse[:-1, :-1] + coarse[1:, :-1] +
                               coarse[:-1, 1:] + coarse[1:, 1:])
    
    return fine


def test_interpolation(N_coarse, N_fine):
    """
    Test the interpolation function by comparing with the exact solution.
    """
    #Generate coarse grid solution
    _, _, coarse_solution, _, _ = setup_test(N_coarse)
    
    #Interpolate to fine grid
    interpolated_solution = interpolation(coarse_solution, N_fine)
    
    #Generate fine grid points
    X_fine, Y_fine, u_exact_fine, _, _ = setup_test(N_fine)
    
    #Calculate the error (difference)
    error = np.abs(interpolated_solution - u_exact_fine)
    
    #Plot the exact solution
    fig = plt.figure(figsize=(12, 6))
    
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X_fine, Y_fine, u_exact_fine, cmap='viridis')
    ax.set_title('Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u_exact(x, y)')
    
    #Plot the error
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_fine, Y_fine, interpolated_solution, cmap='viridis')
    ax2.set_title('Error between Interpolation and Exact Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Error')
    
    plt.tight_layout()
    plt.show()


def mgv(u0, rhs, v, N, nu1, nu2, level, max_level):
    """
    Multigrid V-cycle for the 2D advection-diffusion problem.

    Updated to handle zero residuals during recursion.
    """
    residual_norm = np.linalg.norm(residual(u0, rhs, N, v))
    #print(f"Level {level}: Residual norm = {residual_norm:.2e}")


    if np.linalg.norm(rhs) == 0:
        print(f"Zero residual encountered at level {level}. Returning initial guess.")
        return u0

    if level == max_level:
        #Coarsest level: Use restarted GMRES to solve
        u, _, _ = restarted_gmres_grid(u0, rhs, N, 1/N, v, tol=1e-6, m=10)
        return u
    else:
        #Pre-smoothing
        u = jacobi(u0, rhs, 2/3, N, nu1)

        #Compute residual
        rf = residual(u, rhs, N, v=v)

        #Restrict residual to the coarser grid
        rc = restriction(rf, N)

        #Recursively call mgv on the coarser grid
        ec = mgv(np.zeros_like(rc), rc, v, N // 2, nu1, nu2, level + 1, max_level)

        #Interpolate the correction back to the finer grid
        ef = interpolation(ec, N)

        #Correct the solution
        u = u + ef

        #Post-smoothing
        u = jacobi(u, rhs, 2/3, N, nu2)

        return u

    
    
def test_mgv(N, max_level,nu1, nu2, v=(1,1)):
    """
    Test the multigrid V-cycle and visualize the exact solution, computed solution, and error in 3D.
    """
    
    #Setup grid and right-hand side
    X, Y, exact, f, h = setup_test(N)  #Exact solution and RHS
    u0 = np.zeros_like(f)  #Initial guess

    #Apply multigrid V-cycle
    u_vcycle = mgv(u0, f, v, N, nu1, nu2, level=1, max_level=max_level)

    #Compute error
    error = (u_vcycle - exact)

    #Plot the results in 3D
    fig = plt.figure(figsize=(18, 6))

    #Exact solution
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, exact, cmap="viridis")
    ax1.set_title("Exact Solution")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("u(x, y)")

    #Computed solution
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, u_vcycle, cmap="viridis")
    ax2.set_title("Computed Solution (V-Cycle)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("u(x, y)")

    #Error
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, error, cmap="inferno")
    ax3.set_title("Error (Computed - Exact)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Error Magnitude")

    plt.tight_layout()
    plt.show()

    #Print max error for quantitative assessment
    max_error = np.max(np.abs(error))
    print(f"Max error between computed and exact solution: {max_error:.6e}")


def MGV_iters(u0, rhs, v, N, nu1, nu2, max_level, tol=1e-6, max_iter=100):
    u = u0.copy()  #Initial guess
    res_norms = []  #To track normalized residual norms

    #Compute initial residual
    r = residual(u, rhs, N, v=v)
    r0_norm = np.linalg.norm(r)  #Initial residual norm
    res_norm = 1.0  #Start with normalized residual norm of 1
    res_norms.append(res_norm)
    #print(f"Initial residual norm: {r0_norm:.2e}")

    for iter_num in range(max_iter):
        #Check for convergence
        if res_norm < tol:
            print(f"Converged at iteration {iter_num + 1}: Residual norm = {res_norm:.2e}")
            break

        #Perform one V-cycle
        #print(f"Iteration {iter_num + 1}: Starting V-cycle")
        u = mgv(u, rhs, v, N, nu1, nu2, level=1, max_level=max_level)

        #Compute the new residual and its norm
        r = residual(u, rhs, N, v=v)
        res_norm = np.linalg.norm(r) / r0_norm  # Normalize by r0_norm
        res_norms.append(res_norm)

        #print(f"Iteration {iter_num + 1}: Residual norm = {res_norm:.2e}")

    return u, res_norms



def test_MGV_iters(N, max_level, nu1, nu2, tol, v=(1, 1)):
    #Setup grid and right-hand side
    X, Y, exact, f, h = setup_test(N)  # Exact solution and RHS
    u0 = np.zeros_like(f)  # Initial guess

    #Solve using iterative MGV
    u_mgv, res_norms = MGV_iters(u0, f, v, N, nu1, nu2, max_level, tol, max_iter=200)

    #Compute error
    error = (u_mgv - exact)
    max_error = np.max(error)
    print(f'Max error {max_error}')
    print(f'Last res norm {res_norms[-1]}')

    #Create a 3D plot for the results
    fig = plt.figure(figsize=(18, 6))

    #Exact solution
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, exact, cmap='viridis')
    ax1.set_title("Exact Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u(x, y)")

    #Computed solution
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, u_mgv, cmap='viridis')
    ax2.set_title("Computed Solution (MGV)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u(x, y)")

    #Error surface
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, error, cmap='inferno')
    ax3.set_title("Difference (Computed - Exact)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("Error Magnitude")

    plt.tight_layout()
    plt.show()
