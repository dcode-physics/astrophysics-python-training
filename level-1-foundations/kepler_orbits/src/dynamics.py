import numpy as np
from .constants import ProblemParams

def acceleration(r: np.ndarray, params: ProblemParams) -> np.ndarray:
    '''
    Gravitational acceleration for the Kepler problem.
    
    Parameters:

    r : np.ndarray
        position vector [x, y]

    params : ProblemParams
        Physical parameters (mu = GM)

    Returns:

    np.ndarray
        Acceleration vector [a_x, a_y]
    '''

    r2 = r[0]**2 + r[1]**2
    rmag = np.sqrt(r2)
    return -(params.mu * r) / (rmag**3)