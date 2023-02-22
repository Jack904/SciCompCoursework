import scipy
import matplotlib.pyplot as plt


def ode(t,y,a,b,d): #Keeping t in in case our ode reuires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0])
    dy_dt = b*y[1]*(1-(y[1]/y[0]))

    return [dx_dt, dy_dt]



def shooting(x,F,conds): #Finds the points and time period of the limit cycle
    condition_1 = x[:len(x)-1] - scipy.integrate.solve_ivp(F,[0, x[len(x)-1]],x[:len(x)-1], args=conds, rtol = 1e-6).y[:,-1]
    condition_2 = F(0,x[:len(x)-1],*conds)[0] 

    return [*condition_1,condition_2] # THe * gets rid of the list within



if __name__ == "__main__":
    a =1
    b =0.1
    d =0.1
    condish = [a,b,d]
    # SOlving the ode
    predator = scipy.integrate.solve_ivp(ode,[0, 100],[0.5,0.5],args=condish, rtol = 1e-4)
    #Plotting ODE

    initial_guess = [0.8, 0.2,30]
    result = scipy.optimize.root(shooting, x0 = initial_guess, args=(ode, condish))

    print(result)
    plt.plot(predator.t,predator.y[0,:], label = 'X')
    plt.plot(predator.t,predator.y[1,:], label = 'y')
    plt.plot(result.x[0],result.x[1],marker = 'o')
    plt.legend()
    plt.show()
