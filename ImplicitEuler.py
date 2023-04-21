import numpy as np
import matplotlib.pyplot as plt
from DiagMat import ConstructAandB, Grid, Actual_sol
from IVPODEs import solve_to
import math

def ImplicitEuler(n,a,b,bc_left,bc_right,t_end,initial_cond,dt,D=0.5):

    A, B  = ConstructAandB(n,bc_left,bc_right)
    Gridspace, dx, x = Grid(n,a,b)
    
    t = 0
    no_of_time_steps = math.ceil(t_end/dt)
    
    U_sol = np.zeros((n,no_of_time_steps+1))
    U_sol[0,0] = bc_left
    U_sol[1:-1,0]=initial_cond(x)
    U_sol[-1,0] = bc_right
    
    I = np.identity(n)
    C = (dt*D)/(dx**2)
    i = 0
    while t < t_end:
        
        if t_end-t > dt:
            t += dt
        else:
            t+= (t_end-t)
        i += 1
        u = np.linalg.solve((I-C*A),(U_sol[:,i-1]+C*B))
        
        U_sol[:,i] = u
    return U_sol, Gridspace

def CrankNicholson(n,a,b,bc_left,bc_right,t_end,initial_cond,dt,D=0.5):

    A, B  = ConstructAandB(n,bc_left,bc_right)
    Gridspace, dx, x = Grid(n,a,b)
    
    t = 0
    no_of_time_steps = math.ceil(t_end/dt)
    
    U_sol = np.zeros((n,no_of_time_steps+1))
    U_sol[0,0] = bc_left
    U_sol[1:-1,0]=initial_cond(x)
    U_sol[-1,0] = bc_right
    
    I = np.identity(n)
    C = (dt*D)/(dx**2)
    i = 0
    while t < t_end:
        
        if t_end-t > dt:
            t += dt
        else:
            t+= (t_end-t)
        i += 1
        
        u = np.linalg.solve((I - (C/2)*A),((np.matmul((I + (C/2)*A),(U_sol[:,i-1])))+C*B))
        
        U_sol[:,i] = u
    return U_sol, Gridspace
def RK4PDESolver(n,a,b,bc_left,bc_right,t_end,initial_cond,dt,D=0.5):
    A, B  = ConstructAandB(n,bc_left,bc_right)
    Gridspace, dx, x = Grid(n,a,b)
    
    t = 0
    no_of_time_steps = math.ceil(t_end/dt)
    
    U_sol = np.zeros((n,no_of_time_steps+1))
    U_sol[0,0] = bc_left
    U_sol[1:-1,0]=initial_cond(x)
    U_sol[-1,0] = bc_right

    sol = solve_to(PDE , U_sol[:,0], 0,t_end, dt,'RK4', args = [D, A, B, dx])
    return sol, Gridspace
def InitialCond(x):
    return np.sin(np.pi*x)
def PDE(t,u,D,A_dd,b_dd, dx):
    return D/dx**2 *(A_dd @ u +b_dd)


if __name__ == '__main__':
    N = 1000
    D = 0.1
    a = 0
    b = 1
    bc_left = 0
    bc_right = 0
    t_end = 1
    dt = 0.1
    real_x = np.linspace(0,1,100)
    
    
    
    
    
    # U_Imp,X_Imp = ImplicitEuler(N,a,b,bc_left,bc_right,t_end,InitialCond,dt,D)
    # U_Crank,X_Crank = CrankNicholson(N,a,b,bc_left,bc_right,t_end,InitialCond,dt,D) 
    # real_U = Actual_sol(real_x,1,0,1,0.1)
    # plt.plot(X_Imp,U_Imp[:,-1],'o', label = 'Implicit')
    # plt.plot(X_Crank,U_Crank[:,-1],'o',label = 'Crank Nicholson')
    # plt.plot(real_x,real_U, label = 'Real Solution')
    # plt.legend(loc = 'upper left')
    # plt.show()


    # Specific problem of U(0.5,2)

    U_exact_imp, X = ImplicitEuler(N,a,b,bc_left,bc_right,2,InitialCond,dt,D)
    U_exact_crank, X = CrankNicholson(N,a,b,bc_left,bc_right,2,InitialCond,dt,D) 
    U_exact_rk4, X = RK4PDESolver(N,a,b,bc_left,bc_right,2,InitialCond,dt,D) 
    U_exact_real = np.exp(-0.2*np.pi**2)
    
    print(U_exact_imp[500,-1])
    print(U_exact_crank[500,-1])
    print(U_exact_real)

# Crank Nicholson is closer to the actual value as it is more accurate than the implicit method 
