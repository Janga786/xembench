"""Small pure-math helpers shared by the eval runner and analysis CLIs."""

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials.

    Preferred over the normal approximation because eval cells are small
    (n = 50-100) and success rates sit near 0 or 1 exactly where the normal
    approximation is worst.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)
