import scipy
from Predator_prey import shooting
import numpy as np
import matplotlib.pyplot as plt

def natural_parameter(ode, initial_guess, h, no_of_steps):
    x0 = np.asarray(initial_guess)
    x_vals = []
    y_vals = []
    for i in range(no_of_steps):
        myfunc = lambda u: np.concatenate((ode(u), [u[0] - x0[-1]]))
        result = scipy.optimize.root(myfunc, x0 = x0)
        x_vals.append(result.x[0])
        y_vals.append(result.x[1])
        x0 = x0 - h
    return [x_vals, y_vals]

def funct(x):
    y = x[0]**3 + x[0] -x[1]
    return [y]

def ode(t,y,a=1,b=2,d=3): #Keeping t in in case our ode reuires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0])
    dy_dt = b*y[1]*(1-(y[1]/y[0]))

    return [dx_dt, dy_dt]



#shooting(,ode,[1,2,3])

if __name__ == "__main__":
    [x_vals, y_vals] = natural_parameter(funct,[1,1],0.1,100)
    plt.plot(x_vals,y_vals)
    plt.show()


