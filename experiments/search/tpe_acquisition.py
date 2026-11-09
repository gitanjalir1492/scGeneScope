"""Version-specific Optuna adapter: the only new module using its private API.

Categorical mixture densities are fitted to real observations and evaluated
exactly over caller-supplied candidates. No landscape or ranking is accessible.
"""

import numpy as np

OPTUNA_VERSION = "4.9.0"


def require_compatible_version():
    import optuna

    if optuna.__version__ != OPTUNA_VERSION:
        raise RuntimeError(f"Finite TPE requires Optuna {OPTUNA_VERSION}; found {optuna.__version__}.")


def acquisition(observed, scores, candidates, choices):
    import optuna

    require_compatible_version()
    from optuna.distributions import CategoricalDistribution
    from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
    from optuna.samplers._tpe.sampler import default_gamma

    scores = np.asarray(scores, float)
    if not len(observed) or len(observed) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("TPE requires finite, matched observed configurations and scores.")
    if not candidates:
        raise ValueError("No candidates for TPE acquisition.")
    space = {name: CategoricalDistribution(values) for name, values in choices.items()}

    def encode(rows):
        return {name: np.asarray([dist.to_internal_repr(row[name]) for row in rows], dtype=float)
                for name, dist in space.items()}

    # Sort by objective only to choose membership, then restore observation order
    # so Optuna's chronological weights retain their intended meaning above n=25.
    order = np.argsort(-scores, kind="stable")
    good = set(order[:default_gamma(len(scores))].tolist())
    below = encode([row for index, row in enumerate(observed) if index in good])
    above = encode([row for index, row in enumerate(observed) if index not in good])
    sampler = optuna.samplers.TPESampler(multivariate=True)
    parameters = sampler._parzen_estimator_parameters
    l_density = _ParzenEstimator(below, space, parameters)
    g_density = _ParzenEstimator(above, space, parameters)
    encoded = encode(candidates)
    result = l_density.log_pdf(encoded) - g_density.log_pdf(encoded)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite categorical TPE acquisition.")
    return result
