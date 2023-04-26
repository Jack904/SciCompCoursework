import numpy as np
import matplotlib.pyplot as plt
import math




def euler_step(ode,x_0,t_0,h):
    """
    A function that does one time step solving an ODE using the euler method

    Parameters
    ----------
    ode : function
        The ODE to apply the euler method to. 
    x_0 : int
        An initial point that the step function should use as an input.
    t_0 : int
        The start time that the initial point is evaluated at
    h : int
        The length of time step that you would like to use

    Returns
    -------
    Returns the next point that is output from the euler step
    """
    x_1 = x_0 + h*(ode(t_0,x_0))
    return x_1

def RK4_step(ode,x_0,t_0,h):
    """
    A function that does one time step solving an ODE using the RK4 method

    Parameters
    ----------
    ode : function
        The ODE to apply the RK4 method to. 
    x_0 : int
        An initial point that the step function should use as an input.
    t_0 : int
        The start time that the initial point is evaluated at
    h : int
        The length of time step that you would like to use

    Returns
    -------
    Returns the next point that is output from the RK4 step
    """
    k1 = ode(t_0,x_0)
    k2 = ode(t_0 + h/2,x_0 +h*(k1/2))
    k3 = ode(t_0 + h/2,x_0 +h*(k2/2))
    k4 = ode(t_0 + h,x_0 +h*k3)
    x_1 = x_0 + (1/6)*(k1 +2*k2 +2*k3 +k4)*(h)
    return x_1

def solve_to(ode,t_1,t_2,x_1,deltat_max,ode_solver, args = []):
    """
    A function that solves an initial value ODE.

    Parameters
    ----------
    ode : function
        The ODE that is required to be solved. 
    t_1 : int
        The initial time that the initial point occurs at
    t_2 : int
        The last time point that the user requires to be solved
    x_1 : numpy.array
        The initial value that occurs at t_1
    deltat_max: int
        The maximum step size that the user would like to take
    ode_solver: str
        The solver the user would like to use. This is between 'Euler' and 'RK4'
    args: list
        Any additional arguments that should be input into the ODE

    Returns
    -------
    Returns an array of the points in time that the ODE is solved and an array of 
    the solutions the ODE at these time points
    """
   
   
    h = deltat_max
    max_no_steps = math.ceil(abs(t_2-t_1)/h)
    # Checks if the input x_1 is an int instead of an array.
    # Initialises values
    if type(x_1) == int:
        x_total = np.zeros(max_no_steps+1)
        x_total[0] = x_1
    else:
        x_total = np.zeros((len(x_1),max_no_steps+1))
        for i in range(len(x_1)):
            x_total[0,i] = x_1[i]
    t_total = np.zeros(max_no_steps+1)
    t_total[0] = t_1
    x_n = x_1
    t_n = t_1
    # Checks the output of the ODE to determine its order
    output_check = ode(0,x_1,*args) 
    if type(output_check) == int:
        order = 1
    else:
        order = len(output_check)
        x_n_1 = np.zeros(order)
    if ode_solver == 'Euler':

        for i in range(max_no_steps):
            x_n_1 = x_n + h*np.array(ode(t_n,x_n,*args))
           
            if t_n+h > t_2:
                h = abs(t_n-t_2)
            t_n = t_n + h
            x_n = x_n_1
            x_total[:,i+1] = x_n_1
            t_total[i+1] = t_n
    elif ode_solver == 'RK4':
        for i in range(max_no_steps):  
            k1 = ode(t_n,x_n,*args) 
            k2 = ode(t_n + h/2,x_n + (h/2)*np.array(k1),*args)
            k3 = ode(t_n + h/2,x_n + (h/2)*np.array(k2),*args)
            k4 = ode(t_n + h,x_n + h*np.array(k3), *args)         
            x_n_1 = x_n + (1/6)*(k1 +2*np.array(k2) +2*np.array(k3) +k4)*(h)
            
            if t_n+h > t_2:
                h = abs(t_n-t_2)
            t_n = t_n + h
            x_n=x_n_1
            
            x_total[:,i+1] = x_n_1
            t_total[i+1] = t_n
    return [t_total, x_total]


## ODE Functions to be input
def hopf_ode(t,y): #Keeping t in in case our ode reuires it
    dx_dt = y[0] -y[1] -y[0]*((y[0])**2 +(y[1])**2)
    dy_dt = y[1] +y[0] -y[1]*((y[0])**2 +(y[1])**2)

    return [dx_dt, dy_dt]
def VanDerPol_Ode(t,y,mu):
    dydt = y[1]
    dydt2 = mu*(1 -y[0]**2)*y[1] - y[0]
    return [dydt, dydt2]
def dot_x(t,x):
    return x

if __name__ == '__main__':

    real_x = lambda t: np.exp(t)
    h = 1
    # x_0 = 1
    # t_0 = 0
    # x_1 = x_0 + h*(dot_x(t_0,x_0)) 
    #print(x_1)

    euler_step(dot_x,4,2,1)

    RK4_step(dot_x,4,2,1)  

    err = []
    errRK4 = []
    deltat_max = np.linspace(0.000001,0.0001,100)
    # for i in range(len(deltat_max)):
    #     [x_total, t_total] = solve_to(dot_x,1,0,1,deltat_max[i],'Euler')
    #     error = abs(x_total[-1]-real_x(1))
    #     err.append(error)
    #     #print(x_total[len(x_total)-1])
    # for i in range(len(deltat_max)):
    #     [x_total1, t_total1] = solve_to(dot_x,1,0,1,deltat_max[i],'RK4')
    #     error = abs(x_total1[-1]-real_x(1))
    #     errRK4.append(error)
    #print(x_total1)
    [t_total3, x_total3] = solve_to(dot_x,0,5,1,0.1,'RK4')
    
    #plt.plot(t_total3, x_total3)
    #plt.plot(t_total1, x_total1)
    #plt.plot(t_total,real_x(t_total))
    # plt.loglog(deltat_max,err)
    # plt.loglog(deltat_max,errRK4)
    # plt.xlabel('Deltat Max')
    # plt.ylabel('Error')
    # plt.show()
    [t_total4, x_total4] = solve_to(VanDerPol_Ode,0,10,[0.006,0.006],0.1,'RK4', args = [0.5])
    
    plt.plot(t_total4,x_total4,'o')
    plt.show()
    