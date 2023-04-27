from scipy.sparse import diags
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def ConstructAandB(n, # number of gridpoints
                   bc_left, # left boundary condition
                   bc_right, # right boundary condition
                   bc_left_condition='Dirichlet', # type of boundary condition on left side
                   bc_right_condition = 'Dirichlet', # type of boundary condition on right side
                   dx=0, # grid step size
                   robin_gamma = 0 # the coefficient of the independent variable in the boundary condition
                   ):
    """
    A function that constructs the Add and Bdd matrices to be used to solve
    PDEs

    Parameters
    ----------
    n : int
        Number of gridpoints
    bc_left : int
        The left most boundary condition
    bc_right : int
        The right most boundary condition
    bc_left_condition : str
        The type of boundary condition that the left boundary condition is.
        This can be between 'Dirichlet', 'Neumann' and 'Robin'
    bc_right_condition : str
        The type of boundary condition that the right boundary condition is.
        This can be between 'Dirichlet', 'Neumann' and 'Robin'
    dx : int
        Grid step size
    robin_gamma : int
        This is the coefficient of the independent variable of the robin
        boundary condition of the form
        
        boundary_condition  = bc - robin_gamma*u(b)

        where bc is equal to the input of either bc_Left or bc_right 
        (depending on which condition is being evaluated).
    Returns
    -------
    Returns the Add and Bdd matrices
    """
    if (bc_left_condition == 'Dirichlet') and (bc_right_condition == 'Dirichlet'):
        k = np.ones(n-1)
        A = np.diag(-2*np.ones(n)) + np.diag(k, -1) + np.diag(k, 1)
        B = np.zeros(n)
        B[0] = bc_left
        B[n-1] = bc_right
 
    elif (bc_left_condition == 'Dirichlet') and (bc_right_condition == 'Neumann'):
        k = np.ones(n)
        A = np.diag(-2*np.ones(n+1)) + np.diag(k, -1) + np.diag(k, 1)
        A[n,n-1] = 2
        B = np.zeros(n+1)
        B[0] = bc_left
        B[n] = 2*bc_right*dx
    elif (bc_left_condition == 'Neumann') and (bc_right_condition == 'Dirichlet'):
        k = np.ones(n)
        A = np.diag(-2*np.ones(n+1)) + np.diag(k, -1) + np.diag(k, 1)
        A[n,n-1] = 2
        B = np.zeros(n+1)
        B[0] = bc_left
        B[n] = 2*bc_right*dx

    elif (bc_left_condition == 'Dirichlet') and (bc_right_condition == 'Robin'):
        k = np.ones(n)
        A = np.diag(-2*np.ones(n+1)) + np.diag(k, -1) + np.diag(k, 1)
        A[n,n-1] = 2
        A[n,n] = -2*(1+robin_gamma*dx)
        B = np.zeros(n+1)
        B[0] = bc_left
        B[n] = 2*bc_right*dx
    elif (bc_left_condition == 'Neumann') and (bc_right_condition == 'Neumann'):
        k = np.ones(n+1)
        A = np.diag(-2*np.ones(n+2)) + np.diag(k, -1) + np.diag(k, 1)
        A[0,1] = 2
        A[n,n-1] = 2
        B = np.zeros(n+2)
        B[0] = 2*bc_left*dx
        B[n] = 2*bc_right*dx
 
    elif (bc_left_condition == 'Neumann') and (bc_right_condition == 'Robin'):
        k = np.ones(n+1)
        A = np.diag(-2*np.ones(n+2)) + np.diag(k, -1) + np.diag(k, 1)
        A[0,1] = 2
        A[n,n-1] = 2
        A[n,n] = -2*(1+robin_gamma*dx)
        B = np.zeros(n+2)
        B[0] = 2*bc_left*dx
        B[n] = 2*bc_right*dx
    elif (bc_left_condition == 'Robin') and (bc_right_condition == 'Neumann'):
        k = np.ones(n+1)
        A = np.diag(-2*np.ones(n+2)) + np.diag(k, -1) + np.diag(k, 1)
        A[0,1] = 2
        A[n,n-1] = 2
        A[0,0] = -2*(1+robin_gamma*dx)
        B = np.zeros(n+2)
        B[0] = 2*bc_left*dx
        B[n] = 2*bc_right*dx
    elif (bc_left_condition == 'Robin') and (bc_right_condition == 'Robin'):
        k = np.ones(n+1)
        A = np.diag(-2*np.ones(n+2)) + np.diag(k, -1) + np.diag(k, 1)
        A[0,1] = 2
        A[n,n-1] = 2
        A[0,0] = -2*(1+robin_gamma*dx)
        A[n,n] = -2*(1+robin_gamma*dx)
        B = np.zeros(n+2)
        B[0] = 2*bc_left*dx
        B[n] = 2*bc_right*dx
    return A, B

def Grid(N, # Number of grid points
         a, # initial value
         b, # last value
         bc_left_condition = 'Dirichlet', # type of boundary condition on left side
         bc_right_condition = 'Dirichlet'# type of boundary condition on right side
         ):
    """
    A function that constructs the gridspace required to solve PDEs across

    Parameters
    ----------
    N : int
        Number of gridpoints
    a : int
        The initial x to start the grid upon
    b : int
        The last point to end the grid upon
    bc_left_condition : str
        The type of boundary condition that the left boundary condition is.
        This can be between 'Dirichlet', 'Neumann' and 'Robin'
    bc_right_condition : str
        The type of boundary condition that the right boundary condition is.
        This can be between 'Dirichlet', 'Neumann' and 'Robin'
   
    Returns
    -------
    Returns the Add and Bdd matrices
    """
    if (bc_right_condition == 'Dirichlet') and (bc_left_condition == 'Dirichlet'):
        dx = abs((b-a)/N)
        
        GridSpace = np.linspace(a,b,N)
        x = GridSpace[1:-1]
    elif (bc_right_condition == 'Dirichlet') and (bc_left_condition == 'Neumann'):
        dx = abs((b-a)/N)
        GridSpace = np.linspace(a,b+dx,N+1)
        x = GridSpace[1:-1]
    elif (bc_right_condition == 'Neumann') and (bc_left_condition == 'Dirichlet'):
        dx = abs((b-a)/N)
        GridSpace = np.linspace(a,b+dx,N+1)
        x = GridSpace[1:-1]
    elif (bc_right_condition == 'Dirichlet') and (bc_left_condition == 'Robin'):
        dx = abs((b-a)/N)
        GridSpace = np.linspace(a,b+dx,N+1)
        x = GridSpace[1:-1]
    elif (bc_right_condition == 'Robin') and (bc_left_condition == 'Dirichlet'):
        dx = abs((b-a)/N)
        GridSpace = np.linspace(a,b+dx,N+1)
        x = GridSpace[1:-1]
    elif (bc_right_condition == 'Neumann' or 'Robin') and (bc_left_condition == 'Neumann' or 'Robin'):
        dx = abs((b-a)/N)
        GridSpace = np.linspace(a,b+2*dx,N+2)
        x = GridSpace[1:-1]
    else:
        dx = abs((b-a)/N)
        GridSpace = np.linspace(a,b+dx,N+1)
        x = GridSpace[1:-1]
    return GridSpace, dx,x

## Functions returning real solutions
def DiffusionIC(x,a,b):
    return np.sin((np.pi*(x-a))/(b-a))

def Actual_sol(x,t,a,b,D):
    return (np.exp(-(D*np.pi**2 *t)/(b-a)**2))*(np.sin((np.pi*(x-a))/(b-a)))





