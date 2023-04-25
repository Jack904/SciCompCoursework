from scipy.sparse import diags
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def ConstructAandB(n,bc_left,bc_right,bc_left_condition='Dirichlet',
                   bc_right_condition = 'Dirichlet', dx=0, robin_gamma = 0):
    if bc_left_condition == 'Dirichlet' and bc_right_condition == 'Dirichlet':
        k = np.ones(n-1)
        A = np.diag(-2*np.ones(n)) + np.diag(k, -1) + np.diag(k, 1)
        B = np.zeros(n)
        B[0] = bc_left
        B[n-1] = bc_right
 
    elif bc_left_condition == 'Dirichlet' and bc_right_condition == 'Neumann':
        k = np.ones(n)
        A = np.diag(-2*np.ones(n+1)) + np.diag(k, -1) + np.diag(k, 1)
        A[n,n-1] = 2
        B = np.zeros(n+1)
        B[0] = bc_left
        B[n] = 2*bc_right*dx
 
    elif bc_left_condition == 'Dirichlet' and bc_right_condition == 'Robin':
        k = np.ones(n)
        A = np.diag(-2*np.ones(n+1)) + np.diag(k, -1) + np.diag(k, 1)
        A[n,n-1] = 2
        A[n,n] = -2*(1+robin_gamma*dx)
        B = np.zeros(n+1)
        B[0] = bc_left
        B[n] = 2*bc_right*dx

    return A, B
def q(x):
    return np.ones(np.size(x))
def Grid(N,a,b, bc_right_condition = 'Dirichlet'):
    if bc_right_condition == 'Dirichlet':
        dx = abs((b-a)/N)
        
        GridSpace = np.linspace(a,b,N)
        x = GridSpace[1:-1]
    if bc_right_condition == 'Neumann' or bc_right_condition == 'Robin':
        dx = abs((b-a)/N)
        
        GridSpace = np.linspace(a,b+dx,N+1)
        x = GridSpace[1:-1]
    return GridSpace, dx,x
def PDE(t,u,D,A_dd,b_dd, dx):
    return D/dx**2 *(A_dd @ u +b_dd)
def DiffusionIC(x,a,b):
    return np.sin((np.pi*(x-a))/(b-a))
def InitialCond(f,bc_left, bc_right):
    u = np.zeros(len(f)+2)
    u[0] = bc_left
    for i in range(len(f)):
        u[i+1] = f[i]
    u[-1] = bc_right
    return u
def Actual_sol(x,t,a,b,D):
    return (np.exp(-(D*np.pi**2 *t)/(b-a)**2))*(np.sin((np.pi*(x-a))/(b-a)))

if __name__ == '__main__':

    D = 0.5
    N = 100
    A, B, q = ConstructAandB(N,0,0)

    Gridspace, dx, x = Grid(N,0,1)
    print(dx)
    # u = np.linalg.solve(A,-B - dx**2 * (q(x)))
    sol = solve_ivp(PDE , (0, 1), InitialCond(DiffusionIC(x,0,1),0,0),args = [D, A, B, dx])
    t = sol.t
    u = sol.y


    real_x = np.linspace(0,1,100)
    real_u = Actual_sol(real_x,1,0,1,D)


    plt.plot(Gridspace,u[:,-1],'o',label = 'Numerical')

    plt.plot(real_x,real_u,'o',label = 'Real')
    plt.legend(loc = 'upper left')
    plt.show()




