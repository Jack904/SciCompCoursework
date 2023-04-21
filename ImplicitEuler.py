import numpy as np
import matplotlib.pyplot as plt
from DiagMat import ConstructAandB, Grid
import math

def ImplicitEuler(n,a,b,bc_left,bc_right,t_end,initial_cond,dt,D=0.5):

    A, B  = ConstructAandB(n,bc_left,bc_right)
    Gridspace, dx, x = Grid(n,a,b)
    
    t = 0
    
    U_sol = np.zeros((n+1,n-2))
    
    U_sol[0,:]=initial_cond(x)
    I = np.identity(n)
    C = (dt*D)/(dx**2)
    i = 0
    while t < t_end:
        if t_end-t <dt:
            t += dt
        else:
            t+= (t_end-t)
        i += 1
        
        u = np.linalg.solve((I-C*A),(U_sol[i-1]+C*B))
        print(len(u))
        U_sol[i,:] = u
    return U_sol, Gridspace
def InitialCond(x):
    return np.sin(np.pi*x)



if __name__ == '__main__':
    N = 100
    D = 0.1
    a = 0
    b = 1
    bc_left = 0
    bc_right = 0
    t_end = 1
    dt = 0.1

    
    
    
    
    
    U,X = ImplicitEuler(N,a,b,bc_left,bc_right,t_end,InitialCond,dt,D)

    plt.plot(X,U)
    plt.show


