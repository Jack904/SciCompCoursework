import scipy
import matplotlib.pyplot as plt



def shooting(x, # the values that we iterate through
             F, # the ODE to be used
             conds=[] # any conditions to the ODE
             ): 
    """
    A function that creates the conditions used in numerical
    shooting to find limit cycles of an ODE

    Parameters
    ----------
    x : numpy.array
        The guesses inputted to the conditions from a root finding
        algorithm. Each point in the array should contain a guess
        of the each the independent variables and a guess of the time
        period of the limit cyle. Time period should be at the end
        of the array of the guess
    F : function
        The ODE that shooting is being applied to. The ODE should take
        an input of time, the independent variables and then any
        additional conditions required.
    args : list
        This should contain any of the addition conditions that need to
        be inputted to the ODE

    Returns
    -------
    Returns a function that contains all the conditions needed for a
    limit cycle. This can then be input into a root finding function
    to solve for the conditions
    """

    # Checks to see if there are any extra conditions that need to be
    # input to the ODE so that there are no errors due to no conditions
    # being input
    if conds == []:
        condition_1 = x[:len(x)-1] - scipy.integrate.solve_ivp(F,[0, x[len(x)-1]],
                                                               x[:len(x)-1], rtol = 1e-6).y[:,-1]
        condition_2 = F(0,x[:len(x)-1])[0]
    else:
        
        condition_1 = x[:len(x)-1] - scipy.integrate.solve_ivp(F,[0, x[len(x)-1]],x[:len(x)-1],
                                                                args=(*conds,), rtol = 1e-6).y[:,-1]
        condition_2 = F(0,x[:len(x)-1],*conds)[0] 

    return [*condition_1,condition_2] # THe * gets rid of the list within
def shooting_solve(ode, # the ODE to be used
                   initial_guess, # the initial guess
                   conds = [] # any conditions to the ODE
                   ):
    """
    A function that takes the boundary conditions created in 
    shooting and uses scipy.optimize.root to find a solution
    to the numerical shooting problem

    Parameters
    ----------
    ode : function
        The ODE that shooting is being applied to. The ODE should take
        an input of time, the independent variables and then any
        additional conditions required.
    initial_guess : numpy.array
        An initial guess at the initial values for the limit cycle. 
        This array should start with the guesses at the independent 
        variables and the last value in the array should be a 
        guess for the time period of the limit cycle

    Returns
    -------
    Returns a numpy.array containing the corrected initial values
    for the limit cycle.
    """
    result = scipy.optimize.root(shooting, x0 = initial_guess, args=(ode, conds)).x
    return result


## ODE Functions to be input
def ode(t,y,a,b,d): #Keeping t in in case our ode reuires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0])
    dy_dt = b*y[1]*(1-(y[1]/y[0]))

    return [dx_dt, dy_dt]

if __name__ == "__main__":
    a =1
    b =0.1
    d =0.1
    condish = [a,b,d]
    # SOlving the ode
    predator = scipy.integrate.solve_ivp(ode,[0, 100],[0.5,0.5],args=condish, rtol = 1e-4)
    #Plotting ODE
    initial_guess = [0.8, 0.15,34]
    # result = scipy.optimize.root(shooting, x0 = initial_guess, args=(ode, condish))
    result = shooting_solve(ode, initial_guess, condish)
    shooting()
    # result = shooting_solve(hopf_ode,[0.06, 0.06, 30],[0])
   
    plt.plot(predator.t,predator.y[0,:], label = 'X')
    plt.plot(predator.t,predator.y[1,:], label = 'y')
    plt.plot(predator.t[38],result[1],marker = 'o')
    plt.plot(predator.t[38],result[0],marker = 'o')
    plt.legend()
    plt.show()
