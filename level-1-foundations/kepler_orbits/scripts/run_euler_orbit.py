import numpy as np

from ..src.constants import ProblemParams
from ..src.dynamics import kepler_rhs
from ..src.orbit_utility import initial_conditions_periapsis
from ..src.integrators import integrate_euler

def main():

    params = ProblemParams(mu = 1.0)

    a = 1.0
    e = 0.3

    r0, v0 = initial_conditions_periapsis(a, e, params)
    y0 = np.array([r0[0], r0[1], v0[0], v0[1]])

    t0 = 0.0
    t1 = 20.0 * 2.0 * np.pi  # ~20 orbital periods
    dt = 0.01

    rhs = lambda t, y: kepler_rhs(t, y, params)

    t, y = integrate_euler(rhs, y0, t0, t1, dt)

    r = y[:, 0:2]
    print('Final position:', r[-1])


if __name__ == '__main__':
    main()
