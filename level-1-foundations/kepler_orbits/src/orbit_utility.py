import numpy as np
from .constants import ProblemParams

def initial_conditions_periapsis(a: float, e: float, params: ProblemParams):
    '''
    Initial position and velocity at periapsis for a Kepler orbit.
    
    Parameters
    ----------
    a : float
        Semi-major axis
    e : float    
        Eccentricity (0 <= e < 1)
    params : ProblemParams
        Physical paramters (mu = GM)

    Returns
    ---------
    r0 : np.ndarray
        Initial position [x, y]
    v0 : np.ndarray
        Initial velocity [v_x, v_y]
    '''

    if not (0 <= e < 1):
        raise ValueError('Eccentricity must satisfy 0 <= e < 1')
    
    rp = a * (1.0 - e)
    vp = np.sqrt(params.mu * (1.0 + e) / (a * (1.0 - e)))

    r0 = np.array([rp, 0.0], dtype=float)
    v0 = np.array([0.0, vp], dtype=float)

    return r0, v0


def specific_energy(r: np.ndarray, v: np.ndarray, params: ProblemParams) -> float:
    '''
    Compute specific orbital energy.
    '''

    rmag = np.linalg.norm(r)
    v2 = np.dot(v, v)
    return 0.5 * v2 - params.mu / rmag


def specific_angular_momentum(r: np.ndarray, v: np.ndarray) -> float:
    '''
    z-component of specific angular momentum in 2D.
    '''

    return r[0] * v[1] - r[1] * v[0]