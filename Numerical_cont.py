import scipy
from Predator_prey import shooting
import numpy as np
import matplotlib.pyplot as plt
import pytest # Look into this



def funct(t,x,c):
    return x**3 -x + c


def natural_parameter(ode, initial_point, p0, p1, no_of_steps):
    x0 = [p0,*initial_point]
    
    x_vals = [initial_point]
    c_vals = [p0]
    h = (abs(p0-p1))/no_of_steps
    output_check = ode(0,initial_point,p0)
    for i in range(no_of_steps):
        x0[0] +=  h
        # x0[-1]-=0.25
        if len(output_check) == 1:
            myfunc = lambda u: np.concatenate(([ode(0,u[1:],u[0])],[u[0]-x0[0]]))
        else:
            myfunc = lambda u: np.concatenate(([ode(0,u[1:],u[0])[0]],[ode(0,u[1:],u[0])[1]],[u[0]-x0[0]]))
        result = scipy.optimize.root(myfunc, x0 = x0, tol = 1e-10)
        x0[-1] = result.x[1:]
        

        if result.success == True:
            
            x_vals.append(result.x[1:])
            c_vals.append(result.x[0])
    return [x_vals, c_vals]

def psuedo_parameter(ode, initial_point,p0,p1, no_of_steps, discretisation = lambda x: x):
    h = (abs(p0-p1))/no_of_steps
    if discretisation == 'shooting':
        initial_guess = [*initial_point , 30]
        result = scipy.optimize.root(shooting, x0 = initial_guess, args= (ode, p0))
        xi_minus_one = [p0, *result.x[:-1]]
    else: 
        xi_minus_one = [p0,initial_point]
    [y1, c1]  = natural_parameter(ode, xi_minus_one[1:],p0,p0+h,1)
    xi = [p0+h, *y1[1]]
    x_vals = np.zeros((len(initial_point), no_of_steps+1))
    x_vals[0,0] = xi_minus_one[1]
    x_vals[1,0] = xi_minus_one[2]
    x_vals[0,1] = [*y1[1]][0]
    x_vals[1,1] = [*y1[1]][1]
    c_vals = [p0, p0+h]
    secant = np.zeros(len(xi_minus_one))
    output_check = ode(0,initial_point,p0)
    i =0
    while xi[0] < p1:
        i += 1
        for j in range(len(initial_point)):
            secant[j] = xi[j] - xi_minus_one[j]

        prediction = xi + secant
        if output_check == 1:
            myfunc = lambda u: np.concatenate(([ode(0,u[1:],u[0])], [np.dot((u-prediction),secant)]))
        else:
            myfunc = lambda u: np.concatenate(([ode(0,u[1:],u[0])[0]], [ode(0,u[1:],u[0])[1]], [np.dot((u-prediction),secant)]))
        result = scipy.optimize.root(myfunc, x0 = prediction)
        xi_minus_one = xi
        xi = result.x
        if result.success == True:
            x_vals[0,i+1] = result.x[1]
            x_vals[1,i+1] = result.x[2]
            # x_vals.append(result.x[1:])
            c_vals.append(result.x[0])

        if i > no_of_steps:
            break
    return [c_vals, x_vals]


def ode(t,y,b): #Keeping t in in case our ode reuires it
    dx_dt = b*y[0] -y[1] -y[0]*((y[0])**2 +(y[1])**2)
    dy_dt = b*y[1] +y[0] -y[1]*((y[0])**2 +(y[1])**2)

    return [dx_dt, dy_dt]

#shooting(,ode,[1,2,3])




if __name__ == "__main__":

    # y,x = natural_parameter(funct, 1.521, -2,2,100)
    # px,py = psuedo_parameter(funct, 1.521 ,-2,3, 200)
    # initial_guess = [0.8, 0.2,30]
    # a =1
    # b =0.1
    # d =0.1
    # condish = [a,b,d]
    # y,x = psuedo_parameter(ode, (0.006,0.006), 0,1,10000, discretisation='shooting')
    # print(x[0])
    # # plt.plot(x[0],y,'.',label = 'Y1')
    # plt.plot(x[1],y,'.',label = 'Y2')

    # # # plt.plot(y_true,x_true, label = 'Real' )
    # # plt.plot(x,y, '.' ,label='Natural Continuation')
    # plt.legend(loc = 'upper left')
    # plt.show()




    # Plotting Our hopf Bifurcation and checking if shooting works with it
    predator = scipy.integrate.solve_ivp(ode,[-10, 100],[0.006,0.006],args = [1],rtol = 1e-8)
    plt.plot(predator.t,predator.y[0,:], label = 'U1')
    plt.plot(predator.t,predator.y[1,:], label = 'U2')
    plt.show()

