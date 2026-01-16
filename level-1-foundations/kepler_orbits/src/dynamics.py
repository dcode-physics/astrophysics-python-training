import numpy as np
from .constants import ProblemParams

def acceleration(r: np.ndarray, params: ProblemParams) -> np.ndarray:
    '''
    Gravitational acceleration for the Kepler problem.
    
    Parameters
    ---------
    r : np.ndarray
        position vector [x, y]

    params : ProblemParams
        Physical parameters (mu = GM)

    Returns
    ---------
    np.ndarray
        Acceleration vector [a_x, a_y]
    '''

    r2 = r[0]**2 + r[1]**2
    rmag = np.sqrt(r2)
    return -(params.mu * r) / (rmag**3)


def kepler_rhs(t: float, y: np.ndarray, params: ProblemParams) -> np.ndarray:
    '''
    First-order ODE system for planar Kepler problem

    state vector y = [x, y, v_x, v_y]
    '''

    r = y[0:2]
    v = y[2:4]

    a = acceleration(r, params)

    return np.array([
        v[0],
        v[1],
        a[0],
        a[1],
    ], dtype = float)
