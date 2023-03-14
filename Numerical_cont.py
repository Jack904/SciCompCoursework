import scipy
from Predator_prey import shooting
import numpy as np
import matplotlib.pyplot as plt
import pytest # Look into this

def natural_parameter(ode, initial_guess, h, no_of_steps, discretisation = lambda u:u):
    x0 = np.asarray(initial_guess)
    x_vals = [x0[0]]
    y_vals = [x0[1]]
    for i in range(no_of_steps):
        myfunc = lambda u: np.concatenate(([ode(0,u)], [u[-1] - x0[-1]]))
        result = scipy.optimize.root(myfunc, x0 = x0)
        x0[-1] = x0[-1] - h
        if result.success == True:
            x_vals.append(result.x[0])
            y_vals.append(result.x[1])
    return [x_vals, y_vals]

def psuedo_parameter(ode, initial_guess, h, no_of_steps, discretisation = lambda u:u):
    x0 = np.asarray(initial_guess)
    x_vals = []
    y_vals = []
    for i in range(no_of_steps):
        myfunc = lambda u: np.concatenate((ode(0,u), [u[-1] - x0[-1]]))
        result = scipy.optimize.root(myfunc, x0 = x0)
        x_vals.append(result.x[0])
        y_vals.append(result.x[1])
        x0[-1] = x0[-1] - h
    return [x_vals, y_vals]

def funct(t,x):
    return (x[0]**3) -x[0] +x[1]

def ode(t,y,a=1,b=2,d=3): #Keeping t in in case our ode reuires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0])
    dy_dt = b*y[1]*(1-(y[1]/y[0]))

    return [dx_dt, dy_dt]

#shooting(,ode,[1,2,3])

if __name__ == "__main__":
    [x_vals, y_vals] = natural_parameter(funct,[-10,1000], 10,200)
    x = np.linspace(10,-10,100)
    y = -funct(0,(x,0))
    
    plt.plot(x,y, label = 'Real' )
    plt.plot(x_vals,y_vals, '.' ,label='Continuation')
    plt.legend(loc = 'upper left')
    plt.show()


