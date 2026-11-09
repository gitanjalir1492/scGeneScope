"""Offline analysis only: this module may access the full validation/test table."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_landscape import probability_hit_at_least_one

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
COMPUTE_GRID_POINTS = 101


def wilson(successes, count):
    p, z = successes / count, 1.959963984540054
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def intervals(values, draws):
    """Resample trajectory indices together across all columns; pointwise CIs."""
    if len(values) < 2:
        return np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
    means = np.concatenate([values[batch].mean(axis=1) for batch in np.array_split(draws, 20)])
    return np.percentile(means, [2.5, 97.5], axis=0)


def score_trials(trials, full):
    full = full.set_index("experiment")
    if not full.index.is_unique or not set(trials["experiment"]) <= set(full.index):
        raise ValueError("Invalid offline landscape join.")
    ranked = full["val_accuracy"].sort_values(ascending=False)
    optimum = ranked.iloc[0]
    best = set(ranked.index[ranked == optimum])
    top3 = set(ranked.index[ranked >= ranked.iloc[2]])
    result = []
    for (_, _), sub in trials.groupby(["series", "replicate"], sort=True):
        sub = sub.sort_values("trial").copy()
        sub["full_val_accuracy"] = sub["experiment"].map(full["val_accuracy"])
        sub["runtime_hours"] = sub["experiment"].map(full["runtime_hours"])
        if not np.isfinite(sub["runtime_hours"]).all() or (sub["runtime_hours"] <= 0).any():
            raise ValueError("Compute analysis requires finite positive full-training costs.")
        sub["best_so_far"] = sub["full_val_accuracy"].cummax()
        sub["regret"] = optimum - sub["best_so_far"]
        sub["found_best"] = sub["experiment"].isin(best).cummax()
        sub["found_top3"] = sub["experiment"].isin(top3).cummax()
        sub["cumulative_gpu_hours"] = sub["runtime_hours"].cumsum()
        # Select by *observed* validation only, even for legacy proxy feedback.
        # np.argmax chooses the earliest observed configuration on exact ties.
        observed = sub["val_accuracy"].to_numpy()
        incumbents = [sub.iloc[int(np.argmax(observed[:i + 1]))]["experiment"] for i in range(len(sub))]
        sub["incumbent_experiment"] = incumbents
        for metric in ("test_accuracy", "test_f1"):
            sub[f"incumbent_{metric}"] = [full.loc[name, metric] if metric in full else np.nan for name in incumbents]
        result.append(sub)
    return pd.concat(result, ignore_index=True), optimum, best, top3


def at_compute(trajectories, grid):
    """Only completed evaluations count; carry the final incumbent after stopping."""
    at_cost = np.full((len(trajectories), len(grid)), np.nan)
    hits = np.zeros_like(at_cost)
    for i, trajectory in enumerate(trajectories):
        indices = np.searchsorted(trajectory["cumulative_gpu_hours"].to_numpy(), grid, side="right") - 1
        available = indices >= 0
        at_cost[i, available] = trajectory["best_so_far"].to_numpy()[indices[available]]
        hits[i, available] = trajectory["found_best"].to_numpy()[indices[available]]
    return at_cost, hits


def analyze(frames, full, output, metadata):
    if not frames:
        raise RuntimeError("No matching trajectories for these methods/settings.")
    trials, optimum, best, top3 = score_trials(pd.concat(frames, ignore_index=True), full)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, analysis=dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_bootstrap=N_BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED, confidence_level=0.95,
        compute_grid_points=COMPUTE_GRID_POINTS, intervals="pointwise trajectory bootstrap; Wilson for hits",
    ))
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    trials.to_csv(output / "replay_trials_scored.csv", index=False)
    curves, compute, diagnostic = [], [], []
    grid = np.linspace(0, trials["cumulative_gpu_hours"].max(), COMPUTE_GRID_POINTS)
    for series, sub in trials.groupby("series", sort=True):
        trajectories = [g.sort_values("trial") for _, g in sub.groupby("replicate", sort=True)]
        n, budget = len(trajectories), len(trajectories[0])
        if any(len(g) != budget for g in trajectories):
            raise ValueError("Do not pool trajectories with different budgets.")
        # A series' intervals do not change when unrelated methods are added to a report.
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(n, size=(N_BOOTSTRAP, n))
        matrices = {name: np.stack([g[name].to_numpy(float) for g in trajectories]) for name in
                    ("best_so_far", "regret", "cumulative_gpu_hours", "found_best", "found_top3",
                     "incumbent_test_accuracy", "incumbent_test_f1")}
        bounds = {name: intervals(values, draws) for name, values in matrices.items()
                  if name not in {"found_best", "found_top3"}}
        method = sub["method"].iloc[0]
        for index in range(budget):
            row = dict(series=series, method=method, model=sub["model"].iloc[0], trial=index + 1, n_replicates=n,
                       mean_best_val_accuracy=matrices["best_so_far"][:, index].mean(),
                       sd_best_val_accuracy=matrices["best_so_far"][:, index].std(ddof=1) if n > 1 else np.nan,
                       ci95_low=bounds["best_so_far"][0][index], ci95_high=bounds["best_so_far"][1][index])
            for metric in ("regret", "cumulative_gpu_hours", "incumbent_test_accuracy", "incumbent_test_f1"):
                row[f"mean_{metric}"] = matrices[metric][:, index].mean()
                row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = bounds[metric][0][index], bounds[metric][1][index]
            for metric in ("found_best", "found_top3"):
                successes = matrices[metric][:, index].sum()
                row[f"p_{metric}"] = successes / n
                row[f"p_{metric}_ci95_low"], row[f"p_{metric}_ci95_high"] = wilson(successes, n)
            curves.append(row)
        at_cost, hits = at_compute(trajectories, grid)
        low, high = intervals(at_cost, draws)
        for j, cost in enumerate(grid):
            covered = np.isfinite(at_cost[:, j]).sum()
            # Do not silently compare means over different subsets of trajectories.
            p_low, p_high = wilson(hits[:, j].sum(), n)
            compute.append(dict(series=series, cumulative_gpu_hours=cost, n_replicates=n, n_with_incumbent=covered,
                                mean_best_val_accuracy=at_cost[:, j].mean() if covered == n else np.nan,
                                ci95_low=low[j] if covered == n else np.nan,
                                ci95_high=high[j] if covered == n else np.nan,
                                p_found_best=hits[:, j].mean(), p_found_best_ci95_low=p_low,
                                p_found_best_ci95_high=p_high))
        if method.startswith("tpe_repaired_"):
            fallback = sub["fallback"].astype(str).str.lower().isin(["true", "1"])
            attempts = pd.to_numeric(sub["proposal_attempts"])
            diagnostic.append(dict(series=series, n_evaluations=len(sub), mean_proposal_attempts=attempts.mean(),
                                   rejected_proposals=int((attempts - (~fallback).astype(int)).clip(lower=0).sum()),
                                   n_fallback=int(fallback.sum()), fraction_fallback=fallback.mean()))
    curves, compute = pd.DataFrame(curves), pd.DataFrame(compute)
    curves.to_csv(output / "replay_curves.csv", index=False)
    compute.to_csv(output / "replay_compute_curves.csv", index=False)
    pd.DataFrame(diagnostic, columns=["series", "n_evaluations", "mean_proposal_attempts", "rejected_proposals",
                                     "n_fallback", "fraction_fallback"]).to_csv(output / "replay_tpe_diagnostics.csv", index=False)
    mix = trials.assign(family=trials["experiment"].str.split("/").str[0])
    proportions = pd.crosstab(mix["series"], mix["family"], normalize="index").add_prefix("fraction_")
    proportions["n_distinct_first_picks"] = mix.loc[mix["trial"].eq(1)].groupby("series")["experiment"].nunique()
    first_mm = mix.loc[mix["family"].eq("multimodal")].groupby(["series", "replicate"])["trial"].min()
    proportions["first_multimodal_trial_mean"] = first_mm.groupby("series").mean()
    proportions["replicates_without_multimodal"] = mix.groupby("series")["replicate"].nunique() - first_mm.groupby("series").size().reindex(proportions.index, fill_value=0)
    proportions.to_csv(output / "replay_selection_mix.csv")
    analytical = []
    for method in ("random", "multimodal_random"):
        names = set(full["experiment"])
        if method == "multimodal_random":
            names = {name for name in names if name.startswith("multimodal/")}
        for budget in range(1, min(int(curves["trial"].max()), len(names)) + 1):
            analytical.append(dict(method=method, trial=budget,
                                   p_found_best=probability_hit_at_least_one(len(names), len(names & best), budget),
                                   p_found_top3=probability_hit_at_least_one(len(names), len(names & top3), budget)))
    analytical = pd.DataFrame(analytical)
    analytical.to_csv(output / "replay_random_analytical.csv", index=False)
    plot(curves, compute, analytical, optimum, output)
    print(curves.loc[curves["trial"].eq(curves["trial"].max())].to_string(index=False))
    print(f"Saved {output}")


def plot(curves, compute, analytical, optimum, output):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    cost_fig, cost_ax = plt.subplots(figsize=(9, 6))
    for i, (series, sub) in enumerate(curves.groupby("series", sort=True)):
        colour = plt.get_cmap("tab20")(i % 20)
        x = sub["trial"].to_numpy()
        for ax, metric in zip(axes, ("mean_best_val_accuracy", "mean_regret", "p_found_best", "p_found_top3")):
            ax.plot(x, sub[metric], label=series, color=colour)
            lo, hi = (("ci95_low", "ci95_high") if metric == "mean_best_val_accuracy" else
                      ("regret_ci95_low", "regret_ci95_high") if metric == "mean_regret" else
                      (f"{metric}_ci95_low", f"{metric}_ci95_high"))
            ax.fill_between(x, sub[lo].to_numpy(float), sub[hi].to_numpy(float), color=colour, alpha=0.12)
        cost = compute.loc[compute["series"].eq(series)]
        cost_ax.plot(cost["cumulative_gpu_hours"], cost["mean_best_val_accuracy"], label=series, color=colour)
        cost_ax.fill_between(cost["cumulative_gpu_hours"].to_numpy(), cost["ci95_low"].to_numpy(float),
                             cost["ci95_high"].to_numpy(float), color=colour, alpha=0.12)
    for method, sub in analytical.groupby("method"):
        for ax, metric in zip(axes[2:], ("p_found_best", "p_found_top3")):
            ax.plot(sub["trial"], sub[metric], "--", label=f"{method} exact")
    axes[0].axhline(optimum, color="black", ls="--", lw=1)
    for ax, label in zip(axes, ("Best validation accuracy", "Simple regret", "P(found optimum)", "P(found top three)")):
        ax.set(xlabel="Evaluations", ylabel=label)
        ax.legend(fontsize=6)
    cost_ax.set(xlabel="Cumulative full-training GPU-hours", ylabel="Best validation accuracy at shared cost")
    cost_ax.legend(fontsize=7)
    fig.tight_layout()
    cost_fig.tight_layout()
    fig.savefig(output / "replay_best_validation.png", dpi=200)
    cost_fig.savefig(output / "replay_compute.png", dpi=200)
    plt.close(fig)
    plt.close(cost_fig)
