# Kepler Orbits - Numerical Integration of Planetary Motion

This project numerically integrates the Newtonian two-body problem
and evaluates numerical integrator quality using conservation laws and
timestep convergence.

## Conserved quantities used for validation

The following quantities are monitored to assess numerical stability:

- Specific orbital energy:
  \[
  epsilon = 0.5 * v^2 - mu / r
  \]

- Specific angular momentum (z-component):
  \[
  h = x * v_y - y * v_x
  \]

Relative drift in these quantities is used as a primary accuracy metric.
