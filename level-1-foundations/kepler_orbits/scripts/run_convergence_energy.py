import numpy as np  
import matplotlib.pyplot as plt
from pathlib import Path
import csv

from ..src.constants import ProblemParams
from ..src.dynamics import kepler_rhs
from ..src.orbit_utility import initial_conditions_periapsis, specific_energy
from ..src.integrators import integrate_euler, integrate_rk4

def final_energy_error(integrator, rhs, y0, t0, t1, dt, params):

    time_array, state_array = integrator(rhs, y0, t0, t1, dt)

    positions = state_array[:, 0:2]
    velocities = state_array[:, 2:4]

    energy = np.array([specific_energy(positions[i], velocities[i], params) for i in range(len(time_array))])
    rel_energy_error_final = (energy[-1] - energy[0]) / abs(energy[0])

    return rel_energy_error_final


def main():

    params = ProblemParams(mu = 1.0)
    a = 1.0
    e = 0.3

    r0, v0 = initial_conditions_periapsis(a, e, params)

    initial_state = np.array([r0[0], r0[1], v0[0], v0[1]])

    t0 = 0.0
    t1 = 20.0 * 2.0 *np.pi  # fixed integration time
    rhs = lambda t, y: kepler_rhs(t, y, params)  

    # dt values (smaller -> more accurate)
    dts = np.array([2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3], dtype=float)

    methods = [('Euler', integrate_euler), ('RK4', integrate_rk4)]

    results = {}
    for name, integrator in methods:
        final_errors = []
        for dt in dts:
            error = final_energy_error(integrator, rhs, initial_state, t0, t1, dt, params)
            final_errors.append(error)
        results[name] = np.array(final_errors)

    # Save CSV summary
    tables_dir = Path('kepler_orbits/results/tables')
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / 'convergence_summary.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dt', 'euler_rel_energy_final', 'rk4_rel_energy_final'])
        for i in range(len(dts)):
            writer.writerow([dts[i], results["Euler"][i], results["RK4"][i]])

    # Plot convergence
    figures_dir = Path('kepler_orbits/results/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.loglog(dts, np.abs(results["Euler"]), marker = 'o', label = "Euler")
    plt.loglog(dts, np.abs(results["RK4"]), marker = 'o', label = "RK4")

    plt.xlabel('Timestep (code units)')
    plt.ylabel('|ΔE(T) / |E₀||')
    plt.title('Energy Error Convergence with Timestep')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'convergence_energy_dt.png', dpi = 300)
    plt.close()

    print('Saved:', csv_path)
    print('Saved:', figures_dir / 'convergence_energy_dt.png')

if __name__ == '__main__':
    main()
