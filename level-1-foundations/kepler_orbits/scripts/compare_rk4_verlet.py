import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..src.constants import ProblemParams
from ..src.dynamics import kepler_rhs, acceleration
from ..src.orbit_utility import initial_conditions_periapsis, specific_energy
from ..src.integrators import integrate_rk4, integrate_verlet

def main():

    params = ProblemParams(mu = 1.0)

    a = 1.0
    e = 0.3
    r0, v0 = initial_conditions_periapsis(a, e, params)

    y0 = np.array([r0[0], r0[1], v0[0], v0[1]])

    t0 = 0.0
    t1 = 100.0 * 2.0 * np.pi  # long run
    dt = 0.05

    rhs = lambda t, y: kepler_rhs(t, y, params)
    acc = lambda r: acceleration(r, params)

    # RK4
    t_rk4, y_rk4 = integrate_rk4(rhs, y0, t0, t1, dt)
    r_rk4 = y_rk4[:, :2]
    v_rk4 = y_rk4[:, 2:]
    E_rk4 = np.array([specific_energy(r_rk4[i], v_rk4[i], params) for i in range(len(t_rk4))])
    dE_rk4 = (E_rk4 - E_rk4[0]) / abs(E_rk4[0])

    # Verlet
    t_verlet, r_verlet, v_verlet = integrate_verlet(acc, r0, v0, t0, t1, dt)
    E_verlet = np.array([specific_energy(r_verlet[i], v_verlet[i], params) for i in range(len(t_verlet))])
    dE_verlet = (E_verlet - E_verlet[0]) / abs(E_verlet[0])

    outdir = Path('kepler_orbits/results/figures')
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(t_rk4, dE_rk4, label='RK4')
    plt.plot(t_verlet, dE_verlet, label='Velocity-Verlet')
    plt.xlabel('Time (code units)')
    plt.ylabel('ΔE / |E₀|')
    plt.title('Long-term energy behaviour: RK4 vs Verlet')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'energy_rk4_vs_verlet.png', dpi=300)
    plt.close()

    print('Saved:', outdir / 'energy_rk4_vs_verlet.png')


if __name__ == '__main__':
    main()