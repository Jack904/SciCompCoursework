import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import ceil
from IVPODEs import solve_to

# This is my first attempt at solving PDEs with the explicit euler method
# This was before I had decided to construct matrices to make the process of 
# solving them a lot easier

def ExpPdeSolver(u, alpha, beta, a, b, t_end, N, D=1, C = 0.49, solver = 'euler'):
    dx = b-a/N
    dt = ((dx**2)/D)*(C)
    n_of_t_steps = ceil(t_end/dt)
    x_points = np.linspace(a,b,N+1)
    x_int = x_points[1:-1]
    U = np.zeros((n_of_t_steps+1,N-1))  
    U[0,:] = u(x_int,a,b)
    for n in range(n_of_t_steps):
        if solver == 'euler':
            for i in range(N-1):
                if i == 0:
                    U[n+1, 0] = U[n,0] + C*(alpha - 2*U[n,0] + U[n,1])
                if i > 0 and i< N - 2:
                    U[n+1, i] = U[n,i] + C*(U[n,i+1] - 2*U[n,i] + U[n,i-1])
                if i == N-2:
                    U[n+1, N-2] = U[n,N-2] + C*(beta - 2*U[n,N-2] + U[n,N-3])
        if solver == 'RK4':
            for i in range(N-1):
                if i == 0:
                    U[n+1, 0] = solve_to
    return U, x_int, n_of_t_steps
def InitialCond(x,alpha,beta):
    return np.sin((np.pi*(x-alpha))/(beta-alpha))

if __name__ == '__main__':
    a = 0
    b = 1
    alpha = 0
    beta = 0
    N = 20
    t_end = 1
   
    U,x,N_time = ExpPdeSolver(InitialCond,alpha,beta,a,b,t_end,N)

    fig, ax = plt.subplots()
    ax.set_ylim(0,1)
    ax.set_ylabel(f'$x$')
    ax.set_xlabel(f'$u(x,t)$')
    line, = ax.plot(x,U[0,:])
    def animate(i):
        
        line.set_data((x,U[i,:]))
        return line,
    
    ani = animation.FuncAnimation(fig, animate, frames = range(N_time), blit = True, interval = 100)
    plt.show()
    