import numpy as np
import matplotlib.pyplot as plt
import scipy

def q(t,u,mu):
   return np.exp(mu*u)


def finite_problem_construc(u, dx, N, alpha, beta, q=0, D = 1, args = []):
    F = np.zeros(N-1)
    #print(len(F))
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
def finite_solver(initial_guess, dx, N, alpha, beta, q=0, d=1, args = []):
    sol = scipy.optimize.root(finite_problem_construc,x0 = initial_guess, args = ( dx, N, alpha, beta, q, d, args))
    return sol
def real_sol(x, a, b, alpha, beta):
    return ((-alpha+beta)/(b-a))*(x-a) +alpha


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

