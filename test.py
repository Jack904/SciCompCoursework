from Predator_prey import shooting
import scipy
import matplotlib.pyplot as plt
import math
import numpy as np


o = -1
beta = 1
conds = [beta, o]

def test_ode(t,u,beta,o):
    du1_dt = beta*u[0]-u[1] + o*u[0]*(((u[1])**2) + ((u[0])**2))
    du2_dt = u[0] + beta*u[1] + o*u[1]*(((u[1])**2) + ((u[0])**2))
    du3_dt = -u[2]

    return [du1_dt, du2_dt, du3_dt]

def test_ode_sols(t,beta):
    theta = 2*math.pi
    u1 = (math.sqrt(beta))*(math.cos(t+theta))
    u2 = (math.sqrt(beta))*(math.sin(t+theta))
    u3 = math.exp(-t)

    return [u1, u2, u3, theta]

solution = scipy.integrate.solve_ivp(test_ode,[0,30],[1,0,1],args = [beta,o],rtol = 1e-5)
#Plotting ODE

initial_guess = [1,0,0,6]


result = scipy.optimize.root(shooting, x0 = initial_guess, args = (test_ode, conds), tol = 1e-6)
print(result.x)
print(test_ode_sols(4*math.pi,beta))
print(np.isclose(result.x,test_ode_sols(4*math.pi,beta), rtol = 1e-4, atol = 1e-4))
#Explicit Solutions

#u1 = (beta^(1/2))*(math.cos(t+2*math.pi))


plt.plot(solution.t,solution.y[0,:], label = 'u1')
plt.plot(solution.t,solution.y[1,:], label = 'u2')
plt.plot(solution.t,solution.y[2,:], label = 'u3')
plt.plot(0,result.x[1], marker = 'o')
plt.legend()
plt.show()