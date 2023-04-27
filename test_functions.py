from Numerical_Shooting import shooting, shooting_solve
from IVPODEs import solve_to, hopf_ode
from PDESolvers import *
import scipy
import matplotlib.pyplot as plt
import math
import numpy as np
import pytest

def ode(t,u,beta,o):
    du1_dt = beta*u[0]-u[1] + o*u[0]*(((u[1])**2) + ((u[0])**2))
    du2_dt = u[0] + beta*u[1] + o*u[1]*(((u[1])**2) + ((u[0])**2))
    du3_dt = -u[2]

    return [du1_dt, du2_dt, du3_dt]

def ode_sols(t,beta):
    theta = 2*math.pi
    u1 = (math.sqrt(beta))*(math.cos(t+theta))
    u2 = (math.sqrt(beta))*(math.sin(t+theta))
    u3 = math.exp(-t)
    return [u1, u2, u3]

def test_solve_to_euler():
    [output_t, output_x] = solve_to(ode,0,1,[1,0,1],0.001,'Euler', args = [1,-1])
    print(output_x)
    print(ode_sols(1,1))
    assert np.isclose(output_x[0,-1], ode_sols(1,1)[0], rtol = 1e-3)
    assert np.isclose(output_x[1,-1], ode_sols(1,1)[1], rtol = 1e-3)
    assert np.isclose(output_x[2,-1], ode_sols(1,1)[2], rtol = 1e-3)

def test_solve_to_rk4():
    [output_t, output_x] = solve_to(ode,0,1,[1,0,1],0.001,'RK4', args = [1,-1])
    assert np.isclose(output_x[0,-1], ode_sols(1,1)[0], rtol = 1e-5)
    assert np.isclose(output_x[1,-1], ode_sols(1,1)[1], rtol = 1e-5)
    assert np.isclose(output_x[2,-1], ode_sols(1,1)[2], rtol = 1e-5)
def test_shooting_solve():
    output = shooting_solve(ode,[1,0,1,6], [1,-1])
    assert np.isclose(output[0],ode_sols(4*math.pi,1)[0], atol = 1e-2)
    assert np.isclose(output[1],ode_sols(4*math.pi,1)[1], atol = 1e-2)
    assert np.isclose(output[2],ode_sols(4*math.pi,1)[2], atol = 1e-2)
def test_shooting_solve_time_period():
    output = shooting_solve(ode,[1,0,1,6], [1,-1])
    assert np.isclose(output[3],2*math.pi, rtol = 1e-4)
def test_implicit_euler_dirichlet():
    output,X =  ImplicitEuler(101,0,1,0,0,2,InitialCond,0.01,0.1)
    U_exact_real = np.exp(-0.2*np.pi**2)
    assert np.isclose(output[49,-1],U_exact_real,atol = 1e-2)
def test_crank_nicolson_dirichlet():
    output,X =  CrankNicolson(101,0,1,0,0,2,InitialCond,0.01,0.1)
    U_exact_real = np.exp(-0.2*np.pi**2)
    assert np.isclose(output[49,-1],U_exact_real,atol = 1e-2)
def test_explicit_euler_dirichlet():
    output,X =  EXPEulerPDESolver(101,0,1,0,0,2,InitialCond,0.01,0.1)
    U_exact_real = np.exp(-0.2*np.pi**2)
    assert np.isclose(output[49,-1],U_exact_real,atol = 1e-2)
def test_RK4_dirichlet():
    output,X =  RK4PDESolver(101,0,1,0,0,2,InitialCond,0.01,0.1)
    U_exact_real = np.exp(-0.2*np.pi**2)
    assert np.isclose(output[49,-1],U_exact_real,atol = 1e-2)
def test_explicit_euler_bratu_dirichlet():
    output,X =  EXPEulerPDESolver(101,0,1,0,0,2,InitialCond,0.01,1, q=q)
    U_exact_real = (1/(2*1))*(0.5-0)*(0.5-1)
    assert np.isclose(output[49,-1],U_exact_real,atol = 1e-2)
def test_RK4_bratu_dirichlet():
    output,X =  RK4PDESolver(101,0,1,0,0,2,InitialCond,0.01,1,q = q)
    U_exact_real = (1/(2*1))*(0.5-0)*(0.5-1)
    assert np.isclose(output[49,-1],U_exact_real,atol = 1e-2)
def test_implicit_euler_neumann():
    output,X =  ImplicitEuler(101,0,1,0,0,2,InitialCond,0.01,0.1,bc_right_condition='Neumann')
    assert len(output) == 102
def test_crank_nicolson_neumann():
    output,X =  CrankNicolson(101,0,1,0,0,2,InitialCond,0.01,0.1,bc_right_condition='Neumann')
    assert len(output) == 102
def test_explicit_euler_neumann():
    output,X =  EXPEulerPDESolver(101,0,1,0,0,2,InitialCond,0.01,0.1,bc_right_condition='Neumann')
    assert len(output) == 102
def test_RK4_neumann():
    output,X =  RK4PDESolver(101,0,1,0,0,2,InitialCond,0.01,0.1,bc_right_condition='Neumann')
    assert len(output) == 102
def test_implicit_euler_robin():
    output,X =  ImplicitEuler(101,0,1,0,0,2,InitialCond,0.01,0.1,bc_right_condition='Robin', robin_gamma=1)
    assert len(output) == 102
def test_crank_nicolson_robin():
    output,X =  CrankNicolson(101,0,1,0,0,2,InitialCond,0.01,0.1,bc_right_condition='Robin', robin_gamma= 1)
    assert len(output) == 102
def test_explicit_euler_robin():
    output,X =  EXPEulerPDESolver(101,0,1,0,0,2,InitialCond,0.01,0.1,bc_right_condition='Robin', robin_gamma= 1)
    assert len(output) == 102
def test_RK4_robin():
    output,X =  RK4PDESolver(101,0,1,0,0,2,InitialCond,0.01,0.1,bc_right_condition='Robin', robin_gamma=1)
    assert len(output) == 102
if __name__ == '__main__':
    o = -1
    beta = 1
    conds = [beta, o]

    solution = scipy.integrate.solve_ivp(ode,[0,30],[1,0,1],args = [beta,o],rtol = 1e-5)
    #Plotting ODE

    initial_guess = [1,0,1,6]


    result = scipy.optimize.root(shooting, x0 = initial_guess, args = (ode, conds), tol = 1e-6)
    print(result.x)
    print(ode_sols(0,beta))
    print(np.isclose(result.x[0:3],ode_sols(4*math.pi,beta), rtol = 1e-4, atol = 1e-4))
    #Explicit Solutions

    #u1 = (beta^(1/2))*(math.cos(t+2*math.pi))


    # plt.plot(solution.t,solution.y[0,:], label = 'u1')
    # plt.plot(solution.t,solution.y[1,:], label = 'u2')
    # plt.plot(solution.t,solution.y[2,:], label = 'u3')
    # plt.plot(0,result.x[1], marker = 'o')
    # plt.legend()
    # plt.show()