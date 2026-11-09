"""Synthetic tests only: no real feedback files, network calls, or training.

Run explicitly with: poetry run pytest experiments/search/tests/test_search_baselines.py
"""

from collections import Counter
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
import pandas as pd
import pytest

import baseline_analysis as analysis
import baseline_io
import classical_search as classical
import replay_search as replay
import tpe_acquisition
from search_space import build_experiment_name, validate_configuration


NEW_METHODS = [m for m in classical.METHODS if m not in {"random", "bayesian"}]


def history(configs, scores=None):
    if scores is None:
        scores = [0.2 + i / 100 for i in range(len(configs))]
    return [dict(c, status="completed", val_accuracy=str(y), val_f1=str(y)) for c, y in zip(configs, scores)]


def fake_landscape():
    return {build_experiment_name(c): dict(val_accuracy=0.2 + i / 100, val_f1=0.15 + i / 100, runtime_hours=1 + i / 10)
            for i, c in enumerate(classical.catalogue())}


def test_catalogue_and_encoding():
    configs = classical.catalogue()
    assert len(configs) == len({classical.key(c) for c in configs}) == 51
    assert Counter(c["model_setting"] for c in configs) == {"rna_only": 9, "image_only": 6, "multimodal": 36}
    assert len({tuple(classical.categorical(c).values()) for c in configs}) == 51
    assert classical.encode(configs).shape == (51, 16)
    for config in configs:
        validate_configuration(config)
        assert classical.configuration({k: v or "" for k, v in config.items()}) == config
        encoded = classical.categorical(config)
        assert (encoded["fusion"] == "inactive") == (config["model_setting"] != "multimodal")
        assert (encoded["profile"] == "singleprofile") == (config["aggregation"] is None)


def test_random_protocol_preserved_and_analytical():
    expected = replay.random_search.select_configurations([], 10, 1000)
    result = replay.replay("random", 0, fake_landscape(), 10, 3)
    assert [r["experiment"] for r in result] == [build_experiment_name(c) for c in expected]
    probability = analysis.probability_hit_at_least_one
    assert probability(51, 1, 10) == pytest.approx(10 / 51)
    assert probability(36, 1, 10) == pytest.approx(10 / 36)
    assert probability(51, 3, 10) == pytest.approx(0.48811524609843937)
    assert probability(51, 1, 0) == 0
    assert probability(51, 1, 51) == 1
    assert probability(36, 0, 10) == 0


def test_sampling_only_frequency_check():
    # Fixed seeds and broad tolerance: tests the support/prior, with no evaluations.
    first = Counter(classical.key(classical.random_order("multimodal_random", seed)[0]) for seed in range(1800))
    assert len(first) == 36
    assert all(20 < count < 85 for count in first.values())


def test_stratification_and_exhaustion():
    order = classical.random_order("stratified_random", 123)
    assert len(order) == len({classical.key(c) for c in order}) == 51
    for start in range(0, 18, 3):
        assert {c["model_setting"] for c in order[start:start + 3]} == set(classical.BLOCKS["family"])
    assert sorted(Counter(c["model_setting"] for c in order[:10]).values()) == [3, 3, 4]
    assert order == classical.random_order("stratified_random", 123)


@pytest.mark.parametrize("method", NEW_METHODS)
def test_last_candidate_and_exhaustion(method):
    configs = classical.catalogue(method)
    selected, _ = classical.suggest(history(configs[:-1]), method, 1000, 3)
    assert selected == configs[-1]
    with pytest.raises(ValueError, match="No eligible"):
        classical.suggest(history(configs), method, 1000, 3)


@pytest.mark.parametrize("method", classical.METHODS)
def test_small_synthetic_trajectory_reproducible_and_unique(method):
    first = replay.replay(method, 0, fake_landscape(), 4, 3)
    replay.replay("random", 8, fake_landscape(), 3, 3)
    second = replay.replay(method, 0, fake_landscape(), 4, 3)
    assert first == second
    assert len({r["experiment"] for r in first}) == 4


@pytest.mark.parametrize("method", ["multimodal_random", "multimodal_diverse"])
def test_restricted_budget(method):
    with pytest.raises(ValueError, match="budget"):
        replay.replay(method, 0, fake_landscape(), 37, 3)


def test_common_uniform_start():
    methods = ("tpe_repaired_uniform", "tpe_discrete", "gp_ei", "rf_ei")
    for method in methods:
        rows = []
        for expected in classical.uniform_start(71, 3):
            config, diagnostics = classical.suggest(rows, method, 71, 3)
            assert config == expected
            assert diagnostics["phase"] == "startup"
            rows += history([config])


def test_diversity_ignores_scores_and_maximises_distance():
    configs = classical.catalogue("multimodal_diverse")[:2]
    first, _ = classical.suggest(history(configs, [0.1, 0.9]), "multimodal_diverse", 42, 3)
    second, _ = classical.suggest(history(configs, [0.9, 0.1]), "multimodal_diverse", 42, 3)
    assert first == second
    remaining = [c for c in classical.catalogue("multimodal_diverse") if c not in configs]
    def distance(c):
        return min(sum(classical.categorical(c)[f] != classical.categorical(p)[f] for f in classical.BLOCKS)
                   for p in configs)
    assert distance(first) == max(map(distance, remaining))


def test_repaired_rejections_are_failures_not_observations(monkeypatch):
    config = classical.catalogue()[0]
    real_create = optuna.create_study
    studies = []
    def create(**kwargs):
        study = real_create(**kwargs)
        studies.append(study)
        return study
    monkeypatch.setattr(optuna, "create_study", create)
    monkeypatch.setattr(classical, "MAX_PROPOSALS", 3)
    monkeypatch.setattr(classical.bayesian_search, "suggest_parameters", lambda trial: config)
    candidate, diagnostics = classical.suggest(history([config]), "tpe_repaired_hierarchical", 7, 3)
    assert candidate != config
    assert diagnostics["phase"] == "startup"
    assert diagnostics["fallback"] and diagnostics["proposal_attempts"] == 3
    assert [t.state for t in studies[0].trials] == [optuna.trial.TrialState.COMPLETE] + [optuna.trial.TrialState.FAIL] * 3
    assert all(t.value is None for t in studies[0].trials[1:])


def test_hierarchical_prior_first_draws():
    counts = Counter()
    for seed in range(300):
        candidate, _ = classical.suggest([], "tpe_repaired_hierarchical", seed, 3)
        counts[candidate["model_setting"]] += 1
    assert all(60 < count < 140 for count in counts.values())
    assert len(counts) == 3


def test_optuna_pruned_startup_regression(monkeypatch):
    distribution = optuna.distributions.CategoricalDistribution(["a", "b"])
    random_calls = []
    def random_sample(self, study, trial, name, distribution):
        random_calls.append(name)
        return distribution.choices[0]
    monkeypatch.setattr(optuna.samplers.RandomSampler, "sample_independent", random_sample)
    for state, expected_calls in ((optuna.trial.TrialState.PRUNED, 0), (optuna.trial.TrialState.FAIL, 1)):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=3))
        study.add_trial(optuna.trial.create_trial(params={"x": "a"}, distributions={"x": distribution}, value=0.5))
        for _ in range(2):
            study.add_trial(optuna.trial.create_trial(params={"x": "a"}, distributions={"x": distribution}, state=state))
        random_calls.clear()
        study.ask().suggest_categorical("x", ["a", "b"])
        assert len(random_calls) == expected_calls


def test_discrete_scores_all_unobserved_and_selects_argmax(monkeypatch):
    configs = classical.catalogue()[:3]
    def acquisition(observed, scores, candidates, choices):
        assert observed == [classical.categorical(c) for c in configs]
        assert candidates == [classical.categorical(c) for c in classical.catalogue()[3:]]
        np.testing.assert_allclose(scores, [0.1, 0.2, 0.3])
        return np.arange(len(candidates), dtype=float)
    monkeypatch.setattr(tpe_acquisition, "acquisition", acquisition)
    selected, _ = classical.suggest(history(configs, [0.1, 0.2, 0.3]), "tpe_discrete", 1, 3)
    assert selected == classical.catalogue()[-1]


def test_numbered_ids_are_not_features():
    rows = history(classical.catalogue()[:3])
    before = classical.encode([classical.configuration(row) for row in rows])
    for i, row in enumerate(rows):
        row.update(configuration_id=f"arbitrary_{100 - i}", experiment_id=f"trial_{i}")
    after = classical.encode([classical.configuration(row) for row in rows])
    np.testing.assert_array_equal(before, after)


def test_tpe_single_category_hand_calculation():
    # For one observation, the binary kernel is (3/4, 1/4).
    # Mixing equally with the uniform prior gives (5/8, 3/8).
    values = tpe_acquisition.acquisition([{"x": "a"}, {"x": "b"}], [0.9, 0.1],
                                         [{"x": "a"}, {"x": "b"}], {"x": ("a", "b")})
    np.testing.assert_allclose(values, [np.log(5 / 3), np.log(3 / 5)])


def test_tpe_prior_only_bad_group_and_unequal_group_sizes():
    candidates, choices = [{"x": "a"}, {"x": "b"}], {"x": ("a", "b")}
    # One real observation: g is the uniform prior, l is (5/8, 3/8).
    values = tpe_acquisition.acquisition([{"x": "a"}], [0.9], candidates, choices)
    np.testing.assert_allclose(values, np.log([1.25, 0.75]))
    # Three b observations give g(a)=(3*(1/6)+1/2)/4=1/4.
    values = tpe_acquisition.acquisition([{"x": "a"}] + [{"x": "b"}] * 3,
                                         [0.9, 0.3, 0.2, 0.1], candidates, choices)
    np.testing.assert_allclose(values, np.log([2.5, 0.5]))


def test_tpe_joint_mixture_and_direction():
    candidates = [dict(x=x, y=y) for x in ("a", "b") for y in ("a", "b")]
    observed = [candidates[0], candidates[-1]]
    choices = {"x": ("a", "b"), "y": ("a", "b")}
    values = tpe_acquisition.acquisition(observed, [0.9, 0.1], candidates, choices)
    # Joint mixture: ((.75**2 + .5**2)/2) / ((.25**2 + .5**2)/2) = 13/5.
    np.testing.assert_allclose(values, [np.log(13 / 5), 0, 0, np.log(5 / 13)], atol=1e-12)
    changed = tpe_acquisition.acquisition(observed, [0.1, 0.9], candidates, choices)
    np.testing.assert_allclose(changed, -values, atol=1e-12)
    assert classical.choose_max(values, random.Random(1)) == 0


def test_opaque_id_negative_control_and_version_guard(monkeypatch):
    choices = {"id": ("a", "b", "unseen_1", "unseen_2")}
    values = tpe_acquisition.acquisition([{"id": "a"}, {"id": "b"}], [0.8, 0.2],
                                         [{"id": "unseen_1"}, {"id": "unseen_2"}], choices)
    assert values[0] == pytest.approx(values[1])
    monkeypatch.setattr(optuna, "__version__", "incompatible")
    with pytest.raises(RuntimeError, match="requires Optuna"):
        tpe_acquisition.acquisition([], [], [], choices)


def test_ei_closed_forms():
    result = classical.expected_improvement([0.5, 0.7, 0.3, 0.5], [0.1, 0, 0, 0], 0.5)
    np.testing.assert_allclose(result, [0.1 / np.sqrt(2 * np.pi), 0.2, 0, 0])
    assert classical.expected_improvement(0.5, 0.1, 0.5) == pytest.approx(0.1 / np.sqrt(2 * np.pi))
    with pytest.raises(ValueError):
        classical.expected_improvement([0.5], [-0.1], 0.5)


def test_gp_constant_targets_finite():
    x = classical.encode(classical.catalogue()[:3])
    mean, std, jitter = classical.gp_predictions(x, np.full(3, 0.5), classical.encode(classical.catalogue()[3:6]), 7)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std >= 0).all()
    assert jitter in (1e-8, 1e-6, 1e-4)
    assert np.isfinite(classical.expected_improvement(mean, std, 0.5)).all()


def test_gp_jitter_escalation_and_failure(monkeypatch):
    import sklearn.gaussian_process
    attempted = []
    class Model:
        def __init__(self, **kwargs):
            self.jitter = kwargs["alpha"]
            attempted.append(self.jitter)
        def fit(self, x, y):
            if self.jitter < 1e-4:
                raise np.linalg.LinAlgError("synthetic singular matrix")
        def predict(self, x, return_std):
            return np.zeros(len(x)), np.ones(len(x))
    monkeypatch.setattr(sklearn.gaussian_process, "GaussianProcessRegressor", Model)
    x = np.zeros((2, 2))
    assert classical.gp_predictions(x, [0.1, 0.2], x, 1)[2] == 1e-4
    assert attempted == [1e-8, 1e-6, 1e-4]
    def fail(self, x, y):
        raise np.linalg.LinAlgError("synthetic persistent failure")
    monkeypatch.setattr(Model, "fit", fail)
    with pytest.raises(RuntimeError, match="no silent random fallback"):
        classical.gp_predictions(x, [0.1, 0.2], x, 1)


def test_rf_uses_tree_dispersion(monkeypatch):
    import sklearn.ensemble
    class Tree:
        def __init__(self, value):
            self.value = value
        def predict(self, x):
            return np.full(len(x), self.value)
    class Forest:
        def __init__(self, **kwargs):
            self.estimators_ = [Tree(0.2), Tree(0.8)]
        def fit(self, x, y):
            assert len(x) == len(y) == 2
    monkeypatch.setattr(sklearn.ensemble, "RandomForestRegressor", Forest)
    mean, std = classical.rf_predictions(np.zeros((2, 2)), [0.2, 0.8], np.zeros((1, 2)), 1)
    np.testing.assert_allclose(mean, [0.5])
    np.testing.assert_allclose(std, [np.sqrt(0.18)])
    assert classical.expected_improvement(mean, std, 0.8)[0] > 0


@pytest.mark.parametrize("method", ["gp_ei", "rf_ei"])
def test_surrogate_only_fits_observations(monkeypatch, method):
    configs = classical.catalogue()[:3]
    def predictions(x, y, candidates, seed):
        np.testing.assert_allclose(x, classical.encode(configs))
        np.testing.assert_allclose(y, [0.1, 0.2, 0.3])
        assert len(candidates) == 48
        result = (np.zeros(48), np.ones(48))
        return (*result, 1e-8) if method == "gp_ei" else result
    monkeypatch.setattr(classical, "gp_predictions" if method == "gp_ei" else "rf_predictions", predictions)
    rows = history(configs, [0.1, 0.2, 0.3])
    for row in rows:
        row.update(test_accuracy=999, global_rank=1, unobserved_scores="forbidden")
    state = {}
    replay.propose_classical(rows, method, 1, 3, state)
    for row in rows:
        row.update(test_accuracy=-999, global_rank=51)
    replay.propose_classical(rows, method, 1, 3, state)


def test_proposer_does_not_receive_evaluator(monkeypatch):
    seen = []
    configs = classical.catalogue()
    def propose(**kwargs):
        assert "landscape" not in kwargs
        assert len(kwargs["rows"]) == len(seen)
        assert all("test_accuracy" not in row for row in kwargs["rows"])
        seen.append(kwargs["rows"].copy())
        return configs[len(seen) - 1], "synthetic"
    monkeypatch.setitem(replay.PROPOSERS, "random", propose)
    replay.replay("random", 0, fake_landscape(), 3, 3)
    assert [len(rows) for rows in seen] == [0, 1, 2]


def metadata():
    return dict(budget=3, startup_trials=3, base_seed=1000, landscape="full")


def test_resume_extends_without_truncation(tmp_path):
    calls = []
    def run(*args, **kwargs):
        calls.append(args[1])
        return replay.replay(*args, **kwargs)
    for count in (1, 3, 1):
        baseline_io.run_method(tmp_path, metadata(), "random", count, fake_landscape(), run)
    assert calls == [0, 1, 2]
    saved = pd.read_csv(tmp_path / "random.csv")
    assert len(saved) == 9
    assert saved["replicate"].nunique() == 3


def test_interrupted_replicate_preserves_completed(tmp_path):
    def interrupted(*args, **kwargs):
        if args[1] == 1:
            raise RuntimeError("interrupted")
        return replay.replay(*args, **kwargs)
    with pytest.raises(RuntimeError, match="interrupted"):
        baseline_io.run_method(tmp_path, metadata(), "random", 3, fake_landscape(), interrupted)
    saved = pd.read_csv(tmp_path / "random.csv")
    assert len(saved) == 3 and saved["replicate"].eq(0).all()


def test_resume_refuses_mismatch_and_corruption(tmp_path):
    baseline_io.run_method(tmp_path, metadata(), "random", 1, fake_landscape(), replay.replay)
    before = (tmp_path / "random.csv").read_bytes()
    with pytest.raises(ValueError, match="metadata"):
        baseline_io.run_method(tmp_path, dict(metadata(), budget=4), "random", 2, fake_landscape(), replay.replay)
    assert (tmp_path / "random.csv").read_bytes() == before
    saved = pd.read_csv(tmp_path / "random.csv").iloc[:-1]
    with pytest.raises(ValueError, match="Malformed/incomplete"):
        baseline_io.validate_trials(saved, "random", 3, "full", 1000)


def test_concurrent_writer_refused(tmp_path):
    with baseline_io.writer(tmp_path, metadata()):
        with pytest.raises(RuntimeError, match="Another replay"):
            with baseline_io.writer(tmp_path, metadata()):
                pass


def test_settings_paths_and_metadata(tmp_path):
    source_names = ("classical_search.py", "tpe_acquisition.py", "bayesian_search.py", "random_search.py", "search_space.py")
    for name in source_names:
        (tmp_path / name).write_text("# synthetic source\n")
    landscape = tmp_path / "synthetic.csv"
    landscape.write_text("synthetic feedback\n")
    paths = [baseline_io.protocol(tmp_path, landscape, "full", b, s, seed)[0]
             for b, s, seed in ((10, 3, 1000), (5, 3, 1000), (10, 5, 1000), (10, 3, 2000))]
    assert len(set(paths)) == 4
    assert baseline_io.protocol(tmp_path, landscape, "full", 10, 3, 1000)[0] == paths[0]
    (tmp_path / "classical_search.py").write_text("# changed selector\n")
    assert baseline_io.protocol(tmp_path, landscape, "full", 10, 3, 1000)[0] != paths[0]


def test_offline_test_incumbent_and_compute():
    full = pd.DataFrame(dict(experiment=["a", "b", "c"], val_accuracy=[0.6, 0.9, 0.7],
                             runtime_hours=[1, 2, 3], test_accuracy=[0.8, 0.2, 0.99], test_f1=[0.8, 0.2, 0.99]))
    trials = pd.DataFrame(dict(series=["synthetic"] * 3, replicate=[0] * 3, trial=[1, 2, 3],
                               experiment=["a", "b", "c"], val_accuracy=[0.6, 0.9, 0.7]))
    scored, optimum, _, _ = analysis.score_trials(trials, full)
    assert optimum == 0.9
    assert scored["incumbent_experiment"].tolist() == ["a", "b", "b"]
    assert scored["incumbent_test_accuracy"].tolist() == [0.8, 0.2, 0.2]
    values, hits = analysis.at_compute([scored], np.array([0, 1, 2, 3, 6, 10]))
    np.testing.assert_allclose(values[0], [np.nan, 0.6, 0.6, 0.9, 0.9, 0.9], equal_nan=True)
    np.testing.assert_array_equal(hits[0], [0, 0, 0, 1, 1, 1])
    low, high = analysis.wilson(10, 10)
    assert low < 1 and high == pytest.approx(1)


def test_trajectory_bootstrap_and_compute_coverage():
    values = np.asarray([[0.2, 0.4], [0.8, 1.0]])
    # Enumerate all equally likely two-trajectory bootstrap samples.
    draws = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (10, 1))
    low, high = analysis.intervals(values, draws)
    np.testing.assert_allclose(low, [0.2, 0.4])
    np.testing.assert_allclose(high, [0.8, 1.0])
    first = pd.DataFrame(dict(cumulative_gpu_hours=[1, 3], best_so_far=[0.5, 0.8], found_best=[False, True]))
    second = pd.DataFrame(dict(cumulative_gpu_hours=[2, 4], best_so_far=[0.4, 0.9], found_best=[False, True]))
    at_cost, hits = analysis.at_compute([first, second], np.array([1, 2, 3, 4]))
    assert np.isnan(at_cost[1, 0]) and hits[1, 0] == 0
    np.testing.assert_allclose(at_cost[:, 1:].mean(axis=0), [0.45, 0.6, 0.85])


def test_cli_defaults_preserved():
    args = replay.build_parser().parse_args([])
    assert args.replicates == 100
    assert args.methods == ["random", "bayesian"]
    assert args.budget == 10 and args.startup_trials == 3 and args.landscape == "full"
    assert args.seed == 1000
    assert len(classical.METHODS) == 10
    with pytest.raises(SystemExit):
        replay.build_parser().parse_args(["--run-name", "unsupported"])
