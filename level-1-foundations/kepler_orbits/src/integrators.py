import numpy as np

def euler_step(rhs, t, y, dt):
    '''
    Perform a single forward Euler step
    '''

    return y + dt * rhs(t, y)


def integrate_euler(rhs, y0, t0, t1, dt):
    '''
    Integrate an ODE using forward Euler
    '''

    n_steps = int((t1 - t0) / dt)

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    for n in range(n_steps):
        t[n + 1] = t[n] + dt
        y[n + 1] = euler_step(rhs, t[n], y[n], dt)

    return t, y