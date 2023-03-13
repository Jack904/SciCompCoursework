import numpy as np
import matplotlib.pyplot as plt
import scipy

gamma_1 = 1
gamma_2 = 2
a = 1
b = 2
N = 20

left_boundary = gamma_1
right_boundary = gamma_2
grid = np.linspace(a,b,N-1)
initial_guess = 0.1 *grid
x_ints = grid[1:-1]
dx = (a-b)/N

def finite_solver(u, dx, N, alpha, beta, q=0, D = 1):
    F = np.zeros(N-1)
    #print(len(F))
    F[0] = D*(u[1] - 2*u[0] + alpha)/(dx**2) +q
    for i in range(1,N-2):
        F[i] = D*(u[i+1]-2*u[i]+u[i-1])/(dx**2) + q
    F[N-2] = D*(beta - 2*u[N-2] +u[N-3])/(dx**2) + q
    return F

sol = scipy.optimize.root(finite_solver,x0 = initial_guess, args = ( dx, N, left_boundary, right_boundary))
#print(sol)
def real_sol(x, a, b, alpha, beta):
    return ((-alpha+beta)/(b-a))*(x-a) +alpha
true_y = real_sol(grid,a,b,left_boundary,right_boundary)
def real_sol_q(x, a, b, alpha, beta, D):
    return 
plt.plot(grid, true_y)
plt.plot(grid, sol.x,'.')
plt.show()

