import numpy as np
from ..src.constants import ProblemParams
from ..src.dynamics import acceleration

def test_acceleration_direction_and_magnitude():
    params = ProblemParams(mu = 1.0)

    r = np.array([1.0, 0.0])
    a = acceleration(r, params)

    assert np.allclose(a, np.array([-1.0, 0.0]))