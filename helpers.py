import numpy as np

def sine(x, L):
    '''
    Gives L periodic sine wave
    '''
    return np.sin(2 * np.pi * (1 / L) * x)

def mackey_glass(y):
    pass

def lorenz(y, sig=10, rho=28, beta=8/3):
    f = [sig * (y[1] - y[0]),\
        y[0] * (rho - y[2]) - y[1],\
        y[0] * y[1] - beta * y[2]]
    return np.array(f)

def rossler(y, a=0.2, b=0.2, c=5.7):
    f = [-y[1] - y[2],\
        y[0] + a * y[1],\
        b + y[2] * (y[0] - c)]
    return np.array(f)

def rk4(f, y, dt, params={}):
    '''
    Integrates one step with 4th order Runge-Kutta.
    Args
        - f: Right-hand side of first order system of D.E.s (func takes y, **params)
        - y: State vector
        - params: System parameters (dict)
        - dt: Step size (float)
    Returns
        - x_next: Approximation of the next state (1D array)
    ''' 
    k1 = f(y, **params) * dt
    k2 = f(y + (0.5 * k1), **params) * dt
    k3 = f(y + (0.5 * k2), **params) * dt
    k4 = f(y + k3, **params) * dt
    x_next = y + ((1/6) * (k1 + (2 * k2) + (2 * k3) + k4))
    return x_next