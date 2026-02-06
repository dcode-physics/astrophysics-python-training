import numpy as np
from ..src.constants import ProblemParams
from ..src.dynamics import acceleration

def main():
    params = ProblemParams(mu=1.0)

    # Case 1: r = (1, 0) -> a = (-1, 0)
    r = np.array([1.0, 0.0])
    a = acceleration(r, params)
    print('Case 1: r=(1,0)')
    print('  a =', a)
    assert np.allclose(a, np.array([-1.0, 0.0]), rtol=0, atol=1e-12)

    # Case 2: r = (0, 2) -> |a|=1/4, a = (0, -1/4)
    r = np.array([0.0, 2.0])
    a = acceleration(r, params)
    print('Case 2: r=(0,2)')
    print('  a =', a)
    assert np.allclose(a, np.array([0.0, -0.25]), rtol=0, atol=1e-12)

    # Case 3: inverse-square scaling check
    # r = (1, 0) -> |a| = 1, r = (2, 0) -> |a| = 1/4
    r1 = np.array([1.0, 0.0])
    r2 = np.array([2.0, 0.0])
    a1 = np.linalg.norm(acceleration(r1, params))
    a2 = np.linalg.norm(acceleration(r2, params))
    ratio = a2 / a1
    print('Case 3: scaling |a2| / |a1| =', ratio)
    assert abs(ratio - 0.25) < 1e-12

    print('\nAll acceleration checks passed')

if __name__ == '__main__':
    main()

    