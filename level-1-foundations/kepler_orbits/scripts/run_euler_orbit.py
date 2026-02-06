import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..src.constants import ProblemParams
from ..src.dynamics import kepler_rhs
from ..src.orbit_utility import initial_conditions_periapsis
from ..src.orbit_utility import specific_energy
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
    v = y[:, 2:4]

    energy = np.array([
        specific_energy(r[i], v[i], params)
        for i in range(len(t))
    ])

    energy_relative = (energy - energy[0]) / abs(energy[0])

    print('Final position:', r[-1])
    return t, energy_relative


if __name__ == '__main__':
    t, energy_relative = main()

    outdir = Path('kepler_orbits/results/figures')
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(t, energy_relative)
    plt.xlabel('Time (code units)')
    plt.ylabel('Relative energy error ΔE / |E₀|')
    plt.title('Energy Drift in Forward Euler')
    plt.tight_layout()
    plt.savefig(outdir / 'euler_energy_drift.png', dpi = 300)
    plt.close
