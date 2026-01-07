from dataclasses import dataclass

@dataclass(frozen=True)

class ProblemParams:
    """
    Using non-dimensional paramters for the Kepler problem.
    Choosing units such that mu = GM = 1.
    """
    mu: float = 1.0