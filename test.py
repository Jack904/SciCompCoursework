from Predator_prey import shooting
import scipy
import matplotlib.pyplot as plt


o = -1
beta = 1
conds = [beta, o]

def test_ode(t,u,beta,o):
    du1_dt = beta*u[0]-u[1] + o*u[0]*(((u[1])**2) + ((u[0])**2))
    du2_dt = u[0] + beta*u[1] + o*u[1]*(((u[1])**2) + ((u[0])**2))

    return [du1_dt, du2_dt]

solution = scipy.integrate.solve_ivp(test_ode,[0,30],[0.5,0.5],args = [beta,o],rtol = 1e-5)
#Plotting ODE


initial_guess = [0.7,-0.01,6]


result = scipy.optimize.root(shooting, x0 = initial_guess, args = (test_ode, conds), tol = 1e-6)
print(result.x)

plt.plot(solution.t,solution.y[0,:], label = 'u1')
plt.plot(solution.t,solution.y[1,:], label = 'u2')
plt.plot(result.x[0],result.x[1], marker = 'o')
plt.legend()
plt.show()