import numpy as np
from ..src.constants import ProblemParams
from ..src.orbit_utility import initial_conditions_periapsis, specific_energy

def test_initial_conditions_bound_orbit():
    params = ProblemParams(mu = 1.0)

    r0, v0 = initial_conditions_periapsis(a = 1.0, e = 0.3, params = params)
    energy = specific_energy(r0, v0, params)

    assert energy < 0.0