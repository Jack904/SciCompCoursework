from Predator_prey import shooting
import scipy
import matplotlib.pyplot as plt


o = -1
beta = 1


def test_ode(t,u,beta,o):
    du1_dt = beta*u[0]-u[1] + o*u[0]*(((u[1])**2) + ((u[0])**2))
    du2_dt = u[0] + beta*u[1] + o*u[1]*(((u[1])**2) + ((u[0])**2))

    return [du1_dt, du2_dt]

solution = scipy.integrate.solve_ivp(test_ode,[0,10],[1,1],args = [beta,o],rtol = 1e-5)
#Plotting ODE
plt.plot(solution.t,solution.y[0,:], label = 'X')
plt.plot(solution.t,solution.y[1,:], label = 'y')
plt.plot(1.00040380,-0.000808734428, marker = 'o')
plt.legend()
plt.show()

initial_guess = [0.5,1,6]


result = scipy.optimize.root(shooting, x0 = initial_guess, args = (test_ode, [beta, o]))
print(result)