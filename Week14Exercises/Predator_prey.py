import scipy
import matplotlib.pyplot as plt
a =1
b =0.1
d =0.1

def ode(t,y,a,b,d): #Keeping t in in case our ode reuires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0])
    dy_dt = b*y[1]*(1-(y[1]/y[0]))

    return [dx_dt, dy_dt]

# SOlving the ode
predator = scipy.integrate.solve_ivp(ode,[0, 100],[0.5,0.5],args=[1,0.1,0.1], rtol = 1e-4)
#Plotting ODE
plt.plot(predator.t,predator.y[0,:], label = 'X')
plt.plot(predator.t,predator.y[1,:], label = 'y')
plt.legend()
plt.show()


def shooting(x): #Finds the points and time period of the limit cycle
    condition_1 = x[:2]- scipy.integrate.solve_ivp(ode,[0, x[2]],x[:2],args=[a,b,d], rtol = 1e-4).y[:,-1]
    condition_2 = ode(0,x[:2],a,b,d)[0] 

    return [*condition_1,condition_2] # THe * gets rid of the list within

initial_guess = [0.8, 0.2,30]
result = scipy.optimize.root(shooting, x0 = initial_guess)

print(result.x)
