import scipy
from Predator_prey import shooting
import numpy as np
import matplotlib.pyplot as plt
import pytest # Look into this


def funct(x):
    return (x[1]**3) -x[1] + x[0]


# def natural_parameter(f, x0, p0,p1, no_of_steps):
#     solutions = np.array([x0])
#     c_values = np.array([p0])
#     c_range = np.linspace(p0,p1, no_of_steps)
#     for i in range(0,len(c_range)-1):
#         p = c_range[i]
#         prediction = solutions[-1]
#         result = scipy.optimize.root(f, x0 = prediction,args=(p))
#         if result.success == True:
#             solutions = np.append(solutions,result.x)
#             c_values =  np.append(c_values,p)
#     return c_values[1:], solutions[1:]
def natural_parameter(ode, initial_point, p0, p1, no_of_steps):
    x0 = [p0,initial_point]
    
    x_vals = [initial_point]
    c_vals = [p0]
    h = (abs(p0-p1))/no_of_steps
    for i in range(no_of_steps):
        x0[0] +=  h
        # x0[-1]-=0.25
        myfunc = lambda u: np.concatenate(([ode([x0[0],u[1]])], [u[0] - x0[0]]))
        result = scipy.optimize.root(myfunc, x0 = x0, tol = 1e-10)
        print(result.message)
        
        x0[-1] = result.x[1]
        

        if result.success == True:
            
            x_vals.append(result.x[1])
            c_vals.append(x0[0])
    return [x_vals, c_vals]

def psuedo_parameter(ode, initial_x,p0,p1, no_of_steps):
    x0 = np.asarray(initial_x)
    c_values = np.array(p0)
    c_range = np.linspace(p0,p1, no_of_steps)
    for i in range(no_of_steps):
        secant = x0[i+1]-x0[i]
        predictions = x0[-1] + secant

        myfunc = lambda u: np.concatenate(([ode(u)], [np.dot((u - predictions[-1]),secant)]))
        result = scipy.optimize.root(myfunc, x0 = x0[i])
        np.append(x0,result.x[1])
    
    return [x0]


def ode(t,y,a=1,b=2,d=3): #Keeping t in in case our ode reuires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0])
    dy_dt = b*y[1]*(1-(y[1]/y[0]))

    return [dx_dt, dy_dt]

#shooting(,ode,[1,2,3])




if __name__ == "__main__":
    # x,y = psuedo_parameter(funct,[(-2,1.52137971),(-1.995996,1.52070571)],-2,2,1000)
    x_true = np.linspace(-2,2,100)
    y_true = funct((0,-x_true))
    x,y = natural_parameter(funct, 2, -2,2,1000)
    print(x,y)
    plt.plot(y_true,x_true, label = 'Real' )
    plt.plot(y,x, '.' ,label='Continuation')
    plt.legend(loc = 'upper left')
    plt.show()
 

