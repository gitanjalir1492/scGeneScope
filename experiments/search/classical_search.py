"""Finite-landscape selectors. Inputs contain architecture and observed feedback only.

This module deliberately has no landscape/result-file reader. Configuration order
comes from search_space, never from scores, ranks, or configuration ID numbers.
"""

import random

import numpy as np
from scipy.special import ndtr

import bayesian_search
from search_space import generate_all_configurations, validate_configuration


METHODS = [
    "random", "stratified_random", "multimodal_random", "bayesian",
    "tpe_repaired_hierarchical", "tpe_repaired_uniform", "tpe_discrete",
    "gp_ei", "rf_ei", "multimodal_diverse",
]
RESTRICTED = {"multimodal_random", "multimodal_diverse"}
FIELDS = ("model_setting", "profile_setting", "rna_encoder", "image_encoder", "aggregation", "fusion")
BLOCKS = {
    "family": ("rna_only", "image_only", "multimodal"),
    "rna": ("pca_n2000", "scvi_n200", "scgpt", "inactive"),
    "image": ("resnet50", "imagenet_vit_l", "inactive"),
    "profile": ("singleprofile", "avgpool", "transformer"),
    "fusion": ("concat", "weighted", "inactive"),
}
MAX_PROPOSALS = 255
RF_TREES = 256


def configuration(row):
    config = {key: row.get(key) or None for key in FIELDS}
    validate_configuration(config)
    return config


def key(config):
    return tuple(config.get(name) or None for name in FIELDS)


def catalogue(method=None):
    configs = generate_all_configurations(runnable_only=True)
    if method in RESTRICTED:
        configs = [c for c in configs if c["model_setting"] == "multimodal"]
    return configs


def categorical(config):
    validate_configuration(config)
    return dict(zip(BLOCKS, (
        config["model_setting"], config["rna_encoder"] or "inactive",
        config["image_encoder"] or "inactive",
        "singleprofile" if config["profile_setting"] == "singleprofile" else config["aggregation"],
        config["fusion"] or "inactive",
    )))


def encode(configs):
    # A mismatch in one block contributes one to squared Euclidean distance.
    return np.asarray([
        [float(categorical(c)[name] == value) / np.sqrt(2)
         for name, values in BLOCKS.items() for value in values]
        for c in configs
    ], dtype=float)


def observations(rows):
    configs, scores = [], []
    for row in rows:
        if row.get("status") != "completed":
            raise ValueError("Selectors require completed observations only.")
        configs.append(configuration(row))
        scores.append(float(row["val_accuracy"]))
    if len({key(c) for c in configs}) != len(configs) or not np.isfinite(scores).all():
        raise ValueError("Duplicate configurations or nonfinite observed scores.")
    return configs, np.asarray(scores, dtype=float)


def uniform_start(seed, count):
    # Fixed-length permutation: the startup prefixes agree across BO methods/settings.
    configs = catalogue()
    random.Random(seed).shuffle(configs)
    return configs[:count]


def random_order(method, seed):
    rng = random.Random(seed)
    configs = catalogue(method)
    if method == "multimodal_random":
        rng.shuffle(configs)
        return configs
    pools = {family: [c for c in configs if c["model_setting"] == family]
             for family in BLOCKS["family"]}
    for pool in pools.values():
        rng.shuffle(pool)
    order = []
    while any(pools.values()):
        families = [family for family, pool in pools.items() if pool]
        rng.shuffle(families)
        order.extend(pools[family].pop() for family in families)
    return order


def expected_improvement(mean, std, best):
    """Gaussian EI for maximisation, including the exact zero-variance limit."""
    mean, std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Nonfinite predictive moments or negative standard deviation.")
    delta = mean - best
    z = np.divide(delta, std, out=np.zeros_like(delta), where=std > 0)
    result = np.where(std > 0, delta * ndtr(z) + std * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
                      np.maximum(delta, 0.0))
    return np.maximum(result, 0.0)


def choose_max(values, rng):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite acquisition values.")
    tied = np.flatnonzero(np.isclose(values, values.max(), rtol=1e-12, atol=0.0))
    return int(rng.choice(tied.tolist()))


def repaired_tpe(configs, scores, remaining, seed, startup_trials):
    import optuna

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for config, score in zip(configs, scores):
        params, distributions = bayesian_search.trial_data_for_configuration(config)
        study.add_trial(optuna.trial.create_trial(params=params, distributions=distributions, value=float(score)))
    available = {key(c): c for c in remaining}
    for attempt in range(1, MAX_PROPOSALS + 1):
        trial = study.ask()
        candidate = bayesian_search.suggest_parameters(trial)
        if key(candidate) in available:
            return available[key(candidate)], attempt, False
        # FAIL is excluded by Optuna TPE. PRUNED would contaminate both startup and densities.
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
    rng = random.Random(f"{seed}:fallback:{len(configs)}")
    return rng.choice(remaining), MAX_PROPOSALS, True


def gp_predictions(x, y, candidates, seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    for jitter in (1e-8, 1e-6, 1e-4):
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (0.1, 10.0), nu=2.5),
            alpha=jitter, normalize_y=True, n_restarts_optimizer=3, random_state=seed,
        )
        try:
            model.fit(x, y)
            mean, std = model.predict(candidates, return_std=True)
            if np.isfinite(mean).all() and np.isfinite(std).all():
                return mean, std, jitter
        except np.linalg.LinAlgError:
            continue
    raise RuntimeError("GP failed after all documented jitter levels; no silent random fallback.")


def rf_predictions(x, y, candidates, seed):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=RF_TREES, bootstrap=True, max_features=1.0,
                                  min_samples_leaf=1, random_state=seed, n_jobs=1)
    model.fit(x, y)
    predictions = np.asarray([tree.predict(candidates) for tree in model.estimators_])
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)


def suggest(rows, method, seed, startup_trials):
    """Return a configuration and diagnostics; no evaluator or file paths are accepted."""
    if method not in METHODS or method in {"random", "bayesian"} or startup_trials < 1:
        raise ValueError("Unsupported selector or invalid startup count.")
    configs, scores = observations(rows)
    allowed = catalogue(method)
    allowed_keys = {key(c) for c in allowed}
    used = {key(c) for c in configs}
    if not used <= allowed_keys:
        raise ValueError("History is outside this method's eligible catalogue.")
    remaining = [c for c in allowed if key(c) not in used]
    if not remaining:
        raise ValueError("No eligible unobserved configurations remain.")
    rng = random.Random(f"{seed}:selection:{len(rows)}")
    info = dict(phase="search", proposal_attempts=0, fallback=False, gp_jitter="")
    if len(remaining) == 1:
        info["phase"] = "last_candidate"
        return remaining[0], info
    if method in {"stratified_random", "multimodal_random"}:
        candidate = next(c for c in random_order(method, seed) if key(c) not in used)
        info["phase"] = "random"
    elif method == "multimodal_diverse":
        if not configs:
            candidate = random.Random(seed).choice(remaining)
        else:
            x, previous = encode(remaining), encode(configs)
            distances = ((x[:, None, :] - previous[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            candidate = remaining[choose_max(distances, rng)]
        info["phase"] = "diversity"
    elif len(rows) < startup_trials and method != "tpe_repaired_hierarchical":
        candidate = next(c for c in uniform_start(seed, len(allowed)) if key(c) not in used)
        info["phase"] = "startup"
    elif method.startswith("tpe_repaired_"):
        candidate, attempts, fallback = repaired_tpe(configs, scores, remaining, seed, startup_trials)
        info.update(proposal_attempts=attempts, fallback=fallback,
                    phase="startup" if len(rows) < startup_trials else "tpe")
    elif method == "tpe_discrete":
        from tpe_acquisition import acquisition

        values = acquisition([categorical(c) for c in configs], scores,
                             [categorical(c) for c in remaining], BLOCKS)
        candidate = remaining[choose_max(values, rng)]
        info["phase"] = "tpe"
    elif method in {"gp_ei", "rf_ei"}:
        fit_seed = int(np.random.SeedSequence([seed, len(rows)]).generate_state(1)[0])
        x, candidates = encode(configs), encode(remaining)
        if method == "gp_ei":
            mean, std, jitter = gp_predictions(x, scores, candidates, fit_seed)
            info["gp_jitter"] = jitter
        else:
            mean, std = rf_predictions(x, scores, candidates, fit_seed)
        values = expected_improvement(mean, std, scores.max())
        # If EI underflows everywhere, prefer predictive uncertainty, then seeded ties.
        candidate = remaining[choose_max(std if np.all(values == 0) else values, rng)]
        info["phase"] = "ei"
    else:
        raise ValueError(f"Unsupported classical selector: {method}")
    return candidate, info
