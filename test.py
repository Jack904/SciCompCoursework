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

    u1 = (math.sqrt(beta))*(math.cos(t+2*math.pi))
    u2 = (math.sqrt(beta))*(math.sin(t+2*math.pi))

    return [du1_dt, du2_dt, du3_dt]

def test_ode_sols(t,beta):
    u1 = (math.sqrt(beta))*(math.cos(t+2*math.pi))
    u2 = (math.sqrt(beta))*(math.sin(t+2*math.pi))
    u3 = math.exp(-t)

    return [u1, u2, u3]

solution = scipy.integrate.solve_ivp(test_ode,[0,30],[0.5,0.5,0.5],args = [beta,o],rtol = 1e-5)
#Plotting ODE


initial_guess = [0.8,-0.01,0,6]


result = scipy.optimize.root(shooting, x0 = initial_guess, args = (test_ode, conds), tol = 1e-4)
print(result.x)
print(test_ode_sols(1,beta))

#Explicit Solutions

#u1 = (beta^(1/2))*(math.cos(t+2*math.pi))


plt.plot(solution.t,solution.y[0,:], label = 'u1')
plt.plot(solution.t,solution.y[1,:], label = 'u2')
plt.plot(solution.t,solution.y[2,:], label = 'u3')
plt.plot(result.x[0],result.x[1], marker = 'o')
plt.legend()
plt.show()