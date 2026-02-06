import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..src.constants import ProblemParams
from ..src.dynamics import kepler_rhs
from ..src.orbit_utility import initial_conditions_periapsis, specific_energy
from ..src.integrators import integrate_euler, integrate_rk4

def main():
    
    params = ProblemParams(mu = 1.0)

    a = 1.0
    e = 0.3

    r0, v0 = initial_conditions_periapsis(a, e, params)
    y0 = np.array([r0[0], r0[1], v0[0], v0[1]])

    t0 = 0.0
    t1 = 20.0 * 2.0 * np.pi
    dt = 0.01

    rhs = lambda t, y: kepler_rhs(t, y, params)

    time_euler, state_euler = integrate_euler(rhs, y0, t0, t1, dt)

    time_rk4, state_rk4 = integrate_rk4(rhs, y0, t0, t1, dt)

    position_euler = state_euler[:, 0:2]
    velocity_euler = state_euler[:, 2:4]

    position_rk4 = state_rk4[:, 0:2]
    velocity_rk4 = state_rk4[:, 2:4]

    specific_energy_euler = np.array([specific_energy(position_euler[i], velocity_euler[i], params) for i in range(len(time_euler))])

    specific_energy_rk4 = np.array([specific_energy(position_rk4[i], velocity_rk4[i], params) for i in range(len(time_rk4))])

    relative_energy_error_euler = (specific_energy_euler - specific_energy_euler[0]) / abs(specific_energy_euler[0])

    relative_energy_error_rk4 = (specific_energy_rk4 - specific_energy_rk4[0]) / abs(specific_energy_rk4[0])

    return time_euler, relative_energy_error_euler, time_rk4, relative_energy_error_rk4


if __name__ == '__main__':
    time_euler, relative_energy_error_euler, time_rk4, relative_energy_error_rk4 = main()

    outdir = Path('kepler_orbits/results/figures')
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(time_euler, relative_energy_error_euler, label='Euler')
    plt.plot(time_rk4, relative_energy_error_rk4, label='RK4')
    plt.xlabel('Time (code units)')
    plt.ylabel('Relative Energy Error ΔE / |E₀|')
    plt.title('Energy drift: Euler vs RK4')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'energy_drift_euler_vs_rk4.png', dpi=300)
    plt.close()

    print('Saved:', outdir / 'energy_drift_euler_vs_rk4.png')

