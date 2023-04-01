import scipy
from Predator_prey import shooting
import numpy as np
import matplotlib.pyplot as plt
import pytest # Look into this


def funct(x):
    return (x[1]**3) -x[1] + x[0]


def natural_parameter(ode, initial_point, p0, p1, no_of_steps):
    x0 = [p0,initial_point]
    
    x_vals = [initial_point]
    c_vals = [p0]
    h = (abs(p0-p1))/no_of_steps
    for i in range(no_of_steps):
        x0[0] +=  h
        # x0[-1]-=0.25
        myfunc = lambda u: np.concatenate(([ode([u[0],u[1]])], [u[0] - x0[0]]))
        result = scipy.optimize.root(myfunc, x0 = x0, tol = 1e-10)
        
        x0[-1] = result.x[1]
        

        if result.success == True:
            
            x_vals.append(result.x[1])
            c_vals.append(result.x[0])
    return [x_vals, c_vals]

def psuedo_parameter(ode, initial_point,p0,p1, no_of_steps):
    h = (abs(p0-p1))/no_of_steps
    xi_minus_one = [p0,initial_point]
    [y1, c1]  = natural_parameter(ode, initial_point,p0,p0+h,1)
    xi = [p0+h, y1[1]]
    x_vals = [initial_point, y1[1]]
    c_vals = [p0, p0+h]
    secant = np.zeros(2)
    i =0
    while xi[0] < p1:
        i += 1
        secant[0] = xi[0] - xi_minus_one[0]
        secant[1] = xi[1] - xi_minus_one[1]
        prediction = xi + secant
        myfunc = lambda u: np.concatenate(([ode([u[0],u[1]])], [np.dot((u-prediction),secant)]))
        result = scipy.optimize.root(myfunc, x0 = prediction, method = 'krylov')
        xi_minus_one = xi
        xi = [result.x[0], result.x[1]]
    
        if result.success == True:
            x_vals.append(result.x[1])
            c_vals.append(result.x[0])

        if i > no_of_steps:
            break
    return [c_vals, x_vals]


def ode(t,y,a=1,b=2,d=3): #Keeping t in in case our ode reuires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0])
    dy_dt = b*y[1]*(1-(y[1]/y[0]))

    return [dx_dt, dy_dt]

#shooting(,ode,[1,2,3])




if __name__ == "__main__":
    x_true = np.linspace(-2,2,100)
    y_true = funct((0,-x_true))
    y,x = natural_parameter(funct, 1.52, -2,2,10000)
    px,py = psuedo_parameter(funct, 1.52 ,-2,2, 50)
    initial_guess = [0.8, 0.2,30]
    a =1
    b =0.1
    d =0.1
    condish = [a,b,d]

    result = scipy.optimize.root(shooting, x0 = initial_guess, args=(ode, condish))
    plt.plot(x,y,'.',label = 'Natural cont')
    plt.plot(y_true,x_true, label = 'Real' )
    #plt.plot(x,y, '.' ,label='Natural Continuation')
    plt.legend(loc = 'upper left')
    plt.show()
 

