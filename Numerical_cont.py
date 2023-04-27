import scipy
from Numerical_Shooting import *
import numpy as np
import matplotlib.pyplot as plt
import pytest # Look into this

def natural_parameter(ode, # ODE to be used
                    initial_point, # initial point 
                    p0, # starting parameter value
                    p1, # ending parameter value
                    no_of_steps, # number of steps
                    discretisation = [] # discretisation to be used
                    ):
    """
    A function that uses natural continuation to iteratively solve an 
    ODE problem that is either first or second order.

    Parameters
    ----------
    ode : function
        The ODE that natural parameter continuation is being applied to. The ODE 
        should take an input of time, the independent variables, the parameter
        value that should be changed and then any additional conditions required.
    initial_point : int, numpy.array
        This should contain an initial starting point on the ODE that natural
        parameter continuation should start on
    p0 : int
        The initial parameter value that natural parameter continuation should
        start on
    p1 : int
        The final parameter value that natural parameter continuation should 
        iterate to
    no_of_steps : int
        The number of steps the function should take between p0 and p1
    discretisation : str
        The function needed to discretise the limit cycle. The only option for this
        is 'shooting'

    Returns
    -------
    Returns an array of the parameter values that the function has iterated through
    and an array of each of the solutions that the function has found.
    """
    if discretisation == []:
        #Checks if the initial point is a single number or a list
        if (type(initial_point) != int) and (type(initial_point) != float):
            x0 = [p0,*initial_point]
        else:
            x0 = [p0,initial_point]
        y_vals = [initial_point]
        p_vals = [p0]
        h = (p1-p0)/no_of_steps
        
        output_check = ode(0,initial_point,p0)
        
        if type(output_check) != list:
            output_check = [output_check]
        for i in range(no_of_steps):
            x0[0] +=  h
            
            if len(output_check) == 1:
                myfunc = lambda u: np.concatenate(([ode(0,u[1],x0[0])],[u[0]-x0[0]])) 
            else:
                #This has only been generalised to accept either first order or second order ODES
                myfunc = lambda u: np.concatenate(([ode(0,u[1:],u[0])[0]],[ode(0,u[1:],u[0])[1]],[u[0]-x0[0]]))
            
            
            result = scipy.optimize.root(myfunc, x0 = x0, tol = 1e-10)
            for i in range(len(result.x)):
                if i == 0:
                    pass
                x0[i] = result.x[i] 
            

            if result.success == True:
                
                y_vals.append(result.x[1:])
                p_vals.append(result.x[0])
    elif discretisation == 'shooting':
        x0 = [p0,*initial_point[0:-1]]
        time_period = initial_point[-1]
        y_vals = [initial_point[0:-1]]
        p_vals = [p0]
        h = (p1-p0)/no_of_steps
        
        output_check = ode(0,initial_point,p0)
        for i in range(no_of_steps):
            x0[0] +=  h
            myfunc = lambda u: np.concatenate(([shooting(u[1:],ode,[u[0]])[0]],[shooting(u[1:],ode,[u[0]])[1]],[shooting(u[1:],ode,[u[0]])[2]],[u[0]-x0[0]]))
            
            result = scipy.optimize.root(myfunc, x0 = [*x0, time_period])
            for i in range(len(x0)-1):
                x0[i+1] = result.x[i+1]
            
            time_period = result.x[-1]

            if result.success == True:
                
                y_vals.append([*result.x[1:-1]])
                p_vals.append(result.x[0])
    return [p_vals, y_vals]

def pseudo_parameter(ode, # ODE to be used
                    initial_point, # initial point 
                    p0, # starting parameter value
                    p1, # ending parameter value
                    no_of_steps, # number of steps
                    discretisation = [] # discretisation to be used
                    ):
    """
    A function that uses pseudo arclength continuation to iteratively solve an 
    ODE problem that is either first or second order.

    Parameters
    ----------
    ode : function
        The ODE that pseudo arclength continuation is being applied to. The ODE should take
        an input of time, the independent variables, the parameter value that 
        should be changed and then any additional conditions required.
    initial_point : int, numpy.array
        This should contain an initial starting point on the ODE that pseudo
        arclength continuation should start on
    p0 : int
        The initial parameter value that pseudo arclength continuation should start on
    p1 : int
        The final parameter value that pseudo arclength continuation should iterate to
    no_of_steps : int
        The number of steps the function should take between p0 and p1
    discretisation : str
        The function needed to discretise the limit cycle. The only option for this
        is 'shooting'

    Returns
    -------
    Returns an array of the parameter values that the function has iterated through
    and an array of each of the solutions that the function has found.
    """


    if discretisation == []:
        h = (p1-p0)/no_of_steps
    
        [c1, y1]  = natural_parameter(ode, initial_point,p0,p0+2*h,2)
        output_check = ode(0,initial_point,p0)
        if type(output_check) != list:
            output_check = [output_check]
        
        xi_minus_one = [p0+h, *y1[1][0:len(output_check)]]
        xi = [p0+2*h,*y1[2][0:len(output_check)]]
    
        y_vals = np.zeros((len(output_check), no_of_steps+2))
        p_vals = np.zeros(no_of_steps+2)
        p_vals[0] = p0
        p_vals[1] = p0+h
        for i in range(len(output_check)):
            y_vals[i,0] = xi_minus_one[i+1]
            y_vals[i,1] = [*y1[1]][i]
    
        secant = np.zeros(len(xi_minus_one))

        
        i =0
        while xi[0] < p1:
            i += 1
            
            for j in range(len(xi)):
                secant[j] = xi[j] - xi_minus_one[j] 
            
            prediction = xi + secant
            if len(output_check) == 1:
                myfunc = lambda u: np.concatenate(([ode(0,u[1],u[0])],[np.dot((u-prediction),secant)]))
            else:
                myfunc = lambda u: np.concatenate(([ode(0,u[1:],u[0])[0]],[ode(0,u[1:],u[0])[1]],
                                                   [np.dot((u-prediction),secant)]))
            result = scipy.optimize.root(myfunc, x0 = prediction, tol = 1e-10)
            
            xi_minus_one = xi
            xi = result.x
            if result.success == True:
                for k in range(len(result.x)-1):
                    y_vals[k,i+1] = result.x[k+1]
                p_vals[i+1] = result.x[0]
            
            if i >= no_of_steps:
                break
    elif discretisation == 'shooting':
        h = (p1-p0)/no_of_steps
    
        [c1, y1]  = natural_parameter(ode, initial_point,p0,p0+2*h,2,'shooting')
        output_check = ode(0,initial_point,p0)
        if type(output_check) != list:
            output_check = [output_check]
        
        xi_minus_one = [p0+h, *y1[1][0:len(output_check)]]
        xi = [p0+2*h,*y1[2][0:len(output_check)]]
        
        y_vals = np.zeros((len(output_check), no_of_steps+2))
        p_vals = np.zeros(no_of_steps+2)
        p_vals[0] = p0
        p_vals[1] = p0+h
        for i in range(len(output_check)):
            y_vals[i,0] = xi_minus_one[i+1]
            y_vals[i,1] = [*y1[1]][i]
        time_period = initial_point[-1]
        secant = np.zeros(len(xi_minus_one))

        
        i =0
        while xi[0] < p1:
            i += 1
            for j in range(len(initial_point)-1):
                secant[j] = xi[j] - xi_minus_one[j]
            prediction = xi + secant
            if output_check == 1:
                myfunc = lambda u: np.concatenate(([shooting(u[1:],ode,[u[0]])[0]],
                                                   [ode(0,u[1:-1],u[0])],
                                                   [np.dot((u[0:-1]-prediction),secant)]))
            else:
                myfunc = lambda u: np.concatenate(([shooting(u[1:],ode,[u[0]])[0]],
                                                   [shooting(u[1:],ode,[u[0]])[1]],
                                                   [shooting(u[1:],ode,[u[0]])[2]],
                                                   [np.dot((u[0:-1]-prediction),secant)]))
            
            result = scipy.optimize.root(myfunc, x0 = [*prediction,time_period])
            
            xi_minus_one = xi
            xi = result.x[0:3]
            time_period = result.x[3]
            if result.success == True:
                y_vals[0,i+1] = result.x[1]
                y_vals[1,i+1] = result.x[2]
                # x_vals.append(result.x[1:])
                p_vals[i+1] = result.x[0] 
            if i >= no_of_steps:
                break
            prediction = xi + secant
    return [p_vals, y_vals]


def funct(t,x,c):
    return x**3 -x + c

def hopf_ode(t,y,b): #Keeping t in in case our ode reuires it
    dx_dt = b*y[0] -y[1] -y[0]*((y[0])**2 +(y[1])**2)
    dy_dt = b*y[1] +y[0] -y[1]*((y[0])**2 +(y[1])**2)

    return [dx_dt, dy_dt]


if __name__ == "__main__":
    x,y = natural_parameter(funct, 1.521, -2,2,100)
    px,py = pseudo_parameter(funct, 1.521 ,-2,4, 100)
    # initial_guess = [0.8, 0.2,30]
    # a =1
    # b =0.1
    # d =0.1
    # condish = [a,b,d]
    x1,y1 = pseudo_parameter(hopf_ode, (0.1,0,6), 0, 2, 100, discretisation='shooting')
    # plt.plot(y,x[0][0:len(y)],'o', label = 'Y1')
    # plt.plot(y,x[1][0:len(y)],'o', label = 'Y2')


    # plt.plot(y_true,x_true, label = 'Real' )
    # plt.plot(x,y, 'o' ,label='Natural Continuation')
    # plt.plot(px,py[0,:],'.', label = 'Pseudo Continuation')
    
    plt.plot(x1,y1[0,:],'o',label = 'Y1')
    plt.plot(x1,y1[1,:],'o',label = 'Y2')
    plt.legend(loc = 'upper left')


    # Plotting Our hopf Bifurcation and checking if shooting works with it
    # predator = scipy.integrate.solve_ivp(hopf_ode,[-10, 20],[0.006,0.006],args = [1],rtol = 1e-8)
    # plt.plot(predator.t,predator.y[0,:], label = 'U1')
    # plt.plot(predator.t,predator.y[1,:], label = 'U2')
    plt.show()

