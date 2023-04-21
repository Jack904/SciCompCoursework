from scipy.sparse import diags
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def ConstructAandB(n,bc_left,bc_right):
    k = np.ones(n-1)
    A = np.diag(-2*np.ones(n)) + np.diag(k, -1) + np.diag(k, 1)
    B = np.zeros(n)
    B[0] = bc_left
    B[n-1] = bc_right

    return A, B
def q(x):
    return np.ones(np.size(x))
def Grid(N,a,b):
    dx = abs((b-a)/N)
    print(dx)
    GridSpace = np.linspace(a,b,N)
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
    N = 300
    A, B = ConstructAandB(N,0,0)

    Gridspace, dx, x = Grid(N,0,1)
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




