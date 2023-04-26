import numpy as np
import matplotlib.pyplot as plt
from DiagMat import ConstructAandB, Grid, Actual_sol
from IVPODEs import solve_to
import math
import time

def InitialBoundaryConditions(n, no_of_time_steps, dt,bc_left,bc_right,initial_cond,
                               bc_left_condition= 'Dirichlet', bc_right_condition='Dirichlet'):
    
    if (bc_left_condition == 'Dirichlet') and (bc_right_condition == 'Dirichlet'):
        U_sol = np.zeros((n,no_of_time_steps+1))
        U_sol[0,0] = bc_left
        U_sol[-1,0] = bc_right
        U_sol[1:-1,0]=initial_cond
    elif (bc_left_condition == 'Neumann' or 'Robin') and bc_right_condition == 'Dirichlet':
        U_sol = np.zeros((n+1,no_of_time_steps+1))
        U_sol[0,0] = 2*bc_left*dt
        U_sol[-1,0] = bc_right
        U_sol[1:-1,0]=initial_cond
    elif (bc_right_condition == 'Neumann' or 'Robin') and bc_left_condition == 'Dirichlet':
        U_sol = np.zeros((n+1,no_of_time_steps+1))
        U_sol[-1,0] = 2*bc_right*dt
        U_sol[0,0] = bc_left
        U_sol[1:-1,0]=initial_cond
    elif (bc_right_condition == 'Neumann' or 'Robin') and (bc_left_condition == 'Neumann' or 'Robin'):
        
        U_sol = np.zeros((n+2,no_of_time_steps+1))
        U_sol[-1,0] = 2*bc_right*dt
        U_sol[0,0] = 2*bc_left*dt
        U_sol[1:-1,0]=initial_cond
    return U_sol


def ImplicitEuler(n,a,b,bc_left,bc_right,t_end,initial_cond,dt,D=0.5,
                  bc_left_condition= 'Dirichlet', bc_right_condition='Dirichlet',robin_gamma = 0, q = 0):

    Gridspace, dx, x = Grid(n,a,b,bc_left_condition,bc_right_condition)
    A, B = ConstructAandB(n,bc_left,bc_right,bc_left_condition,bc_right_condition, dx, robin_gamma)
    
    t = 0
    no_of_time_steps = math.ceil(t_end/dt)
    
    
    U_sol = InitialBoundaryConditions(n, no_of_time_steps, dt,bc_left,bc_right,initial_cond(x),
                                      bc_left_condition, bc_right_condition)
    
    I = np.identity(len(A))
    C = (dt*D)/(dx**2)
    i = 0
    while i != no_of_time_steps:
        
        if t_end-t > dt:
            t += dt
        else:
            t+= (t_end-t)
        
        i += 1
    
        
        u = np.linalg.solve((I-C*A),(U_sol[:,i-1]+C*B - (dx**2)*q))


        
        U_sol[:,i] = u
    return U_sol, Gridspace

def CrankNicholson(n,a,b,bc_left,bc_right,t_end,initial_cond,dt,D=0.5,
                  bc_left_condition= 'Dirichlet', bc_right_condition='Dirichlet',robin_gamma = 0, q = 0):

    Gridspace, dx, x = Grid(n,a,b,bc_left_condition,bc_right_condition)
    A, B = ConstructAandB(n,bc_left,bc_right,bc_left_condition,bc_right_condition, dx, robin_gamma)
    t = 0
    no_of_time_steps = math.ceil(t_end/dt)
    
    U_sol = InitialBoundaryConditions(n, no_of_time_steps, dt,bc_left,bc_right,initial_cond(x),
                                      bc_left_condition, bc_right_condition)
    
    I = np.identity(len(A))
    C = (dt*D)/(dx**2)
    i = 0
    while i != no_of_time_steps:
        
        if t_end-t > dt:
            t += dt
        else:
            t+= (t_end-t)
        i += 1
        
        u = np.linalg.solve((I - (C/2)*A),(((I + (C/2)*A)@(U_sol[:,i-1]))+C*B - (dx**2)*q))
        
        U_sol[:,i] = u
    return U_sol, Gridspace
def RK4PDESolver(n,a,b,bc_left,bc_right,t_end,initial_cond,dt,D=0.5,
                  bc_left_condition= 'Dirichlet', bc_right_condition='Dirichlet',robin_gamma = 0, q =0):
    Gridspace, dx, x = Grid(n,a,b,bc_left_condition,bc_right_condition)
    A, B = ConstructAandB(n,bc_left,bc_right,bc_left_condition,bc_right_condition, dx, robin_gamma)
    
    dt =  (dx**2)/2*D
    no_of_time_steps = math.ceil(t_end/dt)
    
    U_sol = InitialBoundaryConditions(n, no_of_time_steps, dt,bc_left,bc_right,initial_cond(x),
                                      bc_left_condition, bc_right_condition)
    
    t_sol, x_sol = solve_to(PDE , 0,t_end,U_sol[:,0], dt,'RK4', args = [D, A, B, dx, q])
    return x_sol, Gridspace
def EXPEulerPDESolver(n,a,b,bc_left,bc_right,t_end,initial_cond,dt,D=0.5,
                  bc_left_condition= 'Dirichlet', bc_right_condition='Dirichlet',robin_gamma = 0, q=0):
    
    Gridspace, dx, x = Grid(n,a,b,bc_left_condition,bc_right_condition)
    A, B= ConstructAandB(n,bc_left,bc_right,bc_left_condition,bc_right_condition, dx, robin_gamma)

    dt =  (dx**2)/2*D
    no_of_time_steps = math.ceil(t_end/dt)
    
    U_sol = InitialBoundaryConditions(n, no_of_time_steps, dt,bc_left,bc_right,initial_cond(x),
                                      bc_left_condition, bc_right_condition)

    t_sol, x_sol = solve_to(PDE , 0,t_end,U_sol[:,0], dt,'Euler', args = [D, A, B, dx, q])
    return x_sol, Gridspace
def InitialCond(x):
    return np.sin(np.pi*x)
def PDE(t,u,D,A_dd,b_dd, dx,q):
    if type(q) == int:
        return D/dx**2 *(A_dd @ u +b_dd - (dx**2)*q)
    else:
        return D/dx**2 *(A_dd @ u +b_dd - (dx**2)*q(0,u))
def q(t,u,mu=1):
    return np.exp(u*mu)


if __name__ == '__main__':
    N = 101
    D = 0.1
    a = 0
    b = 1
    bc_left = 0
    bc_right = 0
    t_end = 1
    dt = 0.001
    real_x = np.linspace(0,1,N)
    
    
    
    
    # U_exact_rk4, X_1 = RK4PDESolver(N,a,b,bc_left,bc_right,t_end,InitialCond,dt,D,'Neumann') 
    # U_exact_Euler, X = EXPEulerPDESolver(N,a,b,bc_left,bc_right,t_end,InitialCond,dt,D,bc_right_condition= 'Robin', robin_gamma=1,bc_left_condition='Neumann') 
    # U_Imp,X_Imp = ImplicitEuler(N,a,b,bc_left,bc_right,t_end,InitialCond,dt,D, bc_right_condition= 'Robin', robin_gamma=1,bc_left_condition='Neumann')
    # U_Crank,X_Crank = CrankNicholson(N,a,b,bc_left,bc_right,t_end,InitialCond,dt,D,bc_right_condition= 'Robin', robin_gamma=1,bc_left_condition='Neumann') 
    # real_U = Actual_sol(real_x,1,0,1,D)
    # plt.plot(X_Imp,U_Imp[:,-1],'o', label = 'Implicit')
    # plt.plot(X_Crank,U_Crank[:,-1],'o',label = 'Crank Nicholson')
    # plt.plot(X, U_exact_Euler[-1],'o',label = 'Euler')
    # plt.plot(X_1, U_exact_rk4[-1],'.',label = 'RK4')
    # # plt.plot(real_x,real_U,'o', label = 'Real Solution')
    # plt.legend(loc = 'upper left')
    # plt.show()


    # Specific problem of U(0.5,2)
    start_imp = time.time()
    U_exact_imp, X = ImplicitEuler(N,a,b,bc_left,bc_right,2,InitialCond,dt,D,)
    end_imp = time.time()
    start_crank = time.time()
    U_exact_crank, X = CrankNicholson(N,a,b,bc_left,bc_right,2,InitialCond,dt,D) 
    end_crank = time.time()
    start_rk4 = time.time()
    U_exact_rk4, X = RK4PDESolver(N,a,b,bc_left,bc_right,2,InitialCond,dt,D) 
    end_rk4 = time.time()
    start_exp = time.time()
    U_exact_Euler, X = EXPEulerPDESolver(N,a,b,bc_left,bc_right,2,InitialCond,dt,D) 
    end_exp = time.time()
    U_exact_real = np.exp(-0.2*np.pi**2)
    
    print('Time taken for Implicit Euler to run was:', end_imp-start_imp) 
    print('Time taken for Crank Nicholson to run was:', end_crank-start_crank) 
    print('Time taken for RK4 to run was:', end_rk4-start_rk4) 
    print('Time taken for Explicit Euler to run was:', end_exp-start_exp) 

# Crank Nicholson is closer to the actual value as it is more accurate than the implicit method 
