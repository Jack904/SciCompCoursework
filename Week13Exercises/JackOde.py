import numpy as np
import matplotlib.pyplot as plt


dot_x =lambda x,t:  x 
real_x = lambda t: np.exp(t)
h = 1
x_0 = 1
t_0 = 0
x_1 = x_0 + h*(dot_x(x_0,t_0)) 
#print(x_1)

def euler_step(ode,x_0,t_0,h):
    x_1 = x_0 + h*(ode(x_0,t_0))
    print(x_1)
euler_step(dot_x,4,2,1)

def RK4_step(ode,x_0,t_0,h):
    k1 = ode(x_0,t_0)
    k2 = ode(x_0 +h*(k1/2),t_0 + h/2)
    k3 = ode(x_0 +h*(k2/2),t_0 + h/2)
    k4 = ode(x_0 +h*k3,t_0 + h)
    x_1 = x_0 + (1/6)*(k1 +2*k2 +2*k3 +k4)*(h)
    print(x_1)
RK4_step(dot_x,4,2,1)    


def solve_to(ode,x_1,t_1,t_2,deltat_max,ode_solver):
    h = deltat_max
    x_total = [x_1]
    t_total = [t_1]
    x_n = x_1
    t_n = t_1
    if ode_solver == 'Euler':

        while t_n < t_2:
            x_n_1 = x_n + h*(ode(x_n,t_n))
            if t_n+h > t_2:
                h = abs(t_n-t_2)
            t_n = t_n + h
            x_n = x_n_1
            x_total.append(x_n)
            t_total.append(t_n)
    elif ode_solver == 'RK4':
        while t_n < t_2:
            k1 = ode(x_n,t_n)
            k2 = ode(x_n +h*(k1/2),t_n + h/2)
            k3 = ode(x_n +h*(k2/2),t_n + h/2)
            k4 = ode(x_n +h*k3,t_n + h)
            x_n_1 = x_n + (1/6)*(k1 +2*k2 +2*k3 +k4)*(h)
            if t_n+h > t_2:
                h = abs(t_n-t_2)
            t_n = t_n + h
            x_n=x_n_1
            x_total.append(x_n)
            t_total.append(t_n)
    return [x_total, t_total]
err = []
deltat_max = np.linspace(1,0.001,1000)
for i in range(len(deltat_max)):
    [x_total, t_total] = solve_to(dot_x,1,0,2,deltat_max[i],'Euler')
    error = abs(x_total[len(x_total)-1]-real_x(t_total[len(t_total)-1]))
    err.append(error)
[x_total3, t_total3] = solve_to(dot_x,1,0,2,0.1,'RK4')
 

#plt.plot(t_total, x_total)
#plt.plot(t_total1, x_total1)
#plt.plot(t_total,real_x(t_total))
plt.loglog(err,deltat_max)
plt.show()
    