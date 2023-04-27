import numpy as np
import matplotlib.pyplot as plt
import scipy

def finite_problem_construc(u, # initial points
                            dx, # grid step size
                            N, # number of grid steps
                            alpha, # left boundary condition
                            beta, # right boundary condition
                            q=0, # source term
                            D = 1, # Diffusion coefficient
                            args = [] # any arguments for the source term
                            ):
    """
    A function that constructs the finite differences conditions that
    can then be solved by a root finding function

    Parameters
    ----------
    u : numpy.array
        The initial values of the problem at the initial time at each
        point in the grid
    dx : int
        The size of the steps between each point
    N : int
        Number of grid steps
    alpha : int
        The left boundary condition of the ODE
    beta : int
        The right boundary condition of the ODE
    q : int, function
        The source term of the equation if there is one. This can be
        either a function or an integer
    D : int
        The Diffusion coefficient of the equation
    args : list
        Any conditions needed to be inputted into the source term

    Returns
    -------
    Returns a function that contains an equation for each grid point
    that is inputted, that is required to solve the ODE.
    """
    F = np.zeros(N-1)
    if type(q) == int:
        F[0] = D*(u[1] - 2*u[0] + alpha)/(dx**2) + q
        for i in range(1,N-2):
            F[i] = D*(u[i+1]-2*u[i]+u[i-1])/(dx**2) + q
        F[N-2] = D*(beta - 2*u[N-2] +u[N-3])/(dx**2) + q
    else:
        
        F[0] = D*(u[1] - 2*u[0] + alpha)/(dx**2) + q(0,u[0],*args)
        for i in range(1,N-2):
            F[i] = D*(u[i+1]-2*u[i]+u[i-1])/(dx**2) + q(0,u[i],*args)
        F[N-2] = D*(beta - 2*u[N-2] +u[N-3])/(dx**2) + q(0,u[N-2],*args)
        
    return F
def finite_solver(initial_points, # initial points
                  dx, # grid step size
                  N, # number of grid steps
                  alpha, # left boundary condition
                  beta, # right boundary condition
                  q=0, # source term
                  D = 1, # Diffusion coefficient
                  args = [] # any arguments for the source term
                  ):
    """
    A function that takes the list of equations constructed by 
    finite_problem_construc and solves it with scipy.optimize.root

    Parameters
    ----------
    initial_points : numpy.array
        The initial values of the problem at the initial time at each
        point in the grid
    dx : int
        The size of the steps between each point
    N : int
        Number of grid steps
    alpha : int
        The left boundary condition of the ODE
    beta : int
        The right boundary condition of the ODE
    q : int, function
        The source term of the equation if there is one. This can be
        either a function or an integer
    D : int
        The Diffusion coefficient of the equation
    args : list
        Any conditions needed to be inputted into the source term

    Returns
    -------
    Returns a function that contains an equation for each grid point
    that is inputted, that is required to solve the ODE.
    """
    sol = scipy.optimize.root(finite_problem_construc,x0 = initial_points, args = ( dx, N, alpha, beta, q, D, args))
    return sol

def real_sol(x, a, b, alpha, beta):
    return ((-alpha+beta)/(b-a))*(x-a) +alpha

def q(t,u,mu):
   return np.exp(mu*u)

if __name__ == '__main__':
    
    a = 0
    b = 1
    N = 20

    left_boundary = 0
    right_boundary = 0
    grid = np.linspace(a,b,N-1)
    initial_guess = 0.1 *grid
    dx = (a-b)/N
    sol = finite_solver(initial_guess, dx, N, left_boundary, right_boundary, q, 1, [0.1])
    #print(sol)

    true_y = real_sol(grid,a,b,left_boundary,right_boundary)

    plt.plot(grid, true_y)
    plt.plot(grid, sol.x,'.')
    plt.show()

