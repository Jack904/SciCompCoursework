from scipy.sparse import diags
import numpy as np
import matplotlib.pyplot as plt


def ConstructAandB(n,bc_left,bc_right):
    k = [np.ones(n-1),-2*np.ones(n),np.ones(n-1)]
    offset = [-1,0,1]
    A = diags(k,offset).toarray()
    B = np.zeros(n)
    B[0] = bc_left
    B[n-1] = bc_right

    return A, B
def q(x):
    return np.ones(np.size(x))
def Grid(N,a,b):
    dx = (a-b)/N
    GridSpace = np.linspace(a,b,N+1)
    x = GridSpace[0:-1]
    return GridSpace,dx,x

A, B = ConstructAandB(10,0,0)
GridSpace, dx, x = Grid(10,0,1)
u = np.linalg.solve(A,-B - dx**2 * (q(x)))

plt.plot(x,u,'o',label = 'Numerical')
plt.show()




