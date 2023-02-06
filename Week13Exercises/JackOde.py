import numpy as np
import matplotlib.pyplot as plt


dot_x =lambda x,t:  x 
h = 1
x_0 = 1
t_0 = 0
x_1 = x_0 + h*(dot_x(x_0,t_0)) 
#print(x_1)

def euler_step(ode,x_0,t_0,h):
    x_1 = x_0 + h*(dot_x(x_0,t_0))
    print(x_1)
euler_step(dot_x,4,2,1)

def solve_to(ode,x_1,t_1,t_2,deltat_max) :
    h = deltat_max
    x_total = [x_1]
    t_total = [t_1]
    x_n = x_1
    t_n = t_1
    i = 0
    while t_n < t_2:
        x_n = x_n + h*(ode(x_n,t_n))
        if t_n+h > t_2:
            h = abs(t_n-t_2)
        t_n = t_n + h
        x_total.append(x_n)
        t_total.append(t_n)
    return [x_total, t_total]
[x_total, t_total] = solve_to(dot_x,1,0,2,0.1)
print(x_total)    
plt.plot(t_total, x_total)
plt.show()
    