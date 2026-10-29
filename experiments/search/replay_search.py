"""Replay the search methods many times on the completed 51-configuration landscape (lookup instead of training).

Each replicate starts from an empty history; when a method picks a configuration, its validation accuracy/F1 are
looked up from a landscape results file instead of being trained. Methods reuse the selection code of the live
search (random_search, bayesian_search, llm_search, summarize_experiment). LLM methods need OpenAI credentials.

    python replay_search.py --methods random bayesian --replicates 200
    python replay_search.py --methods llm_metrics llm_summary --replicates 5
    python replay_search.py --analyze-only
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

import bayesian_search
import llm_search
import random_search
import summarize_experiment
from search_space import build_experiment_name, generate_all_configurations

SEARCH_DIR = Path(__file__).resolve().parent
LANDSCAPES = {
    "full": SEARCH_DIR / "results" / "full_landscape_results.csv",
    "proxy": SEARCH_DIR / "results" / "proxy_landscape_results.csv",
}
REPLAY_DIR = SEARCH_DIR / "results" / "lookup_replays"
OUTPUT_DIR = SEARCH_DIR / "analysis" / "lookup_replay"
# Prompt variants: shuffle = random candidate order and IDs per replicate; budget = state experiment k of N and
# make best-found accuracy the only objective; isolation = keep the one-factor-at-a-time / complexity principles
LLM_VARIANTS = {
    "llm_metrics": dict(mode="metrics", shuffle=False, budget=False, isolation=True),
    "llm_summary": dict(mode="summary", shuffle=False, budget=False, isolation=True),
    "llm_metrics_shuffled": dict(mode="metrics", shuffle=True, budget=False, isolation=True),
    "llm_metrics_budget": dict(mode="metrics", shuffle=False, budget=True, isolation=True),
    "llm_metrics_no_isolation": dict(mode="metrics", shuffle=False, budget=False, isolation=False),
    "llm_metrics_combined": dict(mode="metrics", shuffle=True, budget=True, isolation=False),
}
METHOD_ORDER = ["random", "bayesian", *LLM_VARIANTS]
COLOURS = {"random": "#7f7f7f", "bayesian": "#1f77b4", "llm_metrics": "#d62728", "llm_summary": "#2ca02c",
           "llm_metrics_shuffled": "#ff7f0e", "llm_metrics_budget": "#9467bd",
           "llm_metrics_no_isolation": "#8c564b", "llm_metrics_combined": "#e377c2"}
MAX_LLM_ATTEMPTS = 3
N_BOOTSTRAP = 2000


def load_landscape(name: str) -> dict[str, dict]:
    table = pd.read_csv(LANDSCAPES[name])
    table = table.loc[table["status"].eq("completed")]
    if "runtime_hours" not in table:
        table["runtime_hours"] = np.nan
    expected = {build_experiment_name(config) for config in generate_all_configurations(runnable_only=True)}
    missing = expected - set(table["experiment"])
    if missing:
        raise RuntimeError(f"{name} landscape is missing {len(missing)} configurations: {sorted(missing)[:3]}")
    return table.set_index("experiment")[["val_accuracy", "val_f1", "runtime_hours"]].to_dict("index")


def base_row(config: dict, method: str, number: int, reason: str) -> dict[str, str]:
    # LLM prompt builders filter history on the original method name, so variants share it
    search_method = llm_search.method_name(LLM_VARIANTS[method]["mode"]) if method in LLM_VARIANTS else method
    return {
        "experiment_id": f"{method}_{number:03d}", "search_method": search_method,
        **{key: config[key] or "" for key in
           ["model_setting", "profile_setting", "rna_encoder", "image_encoder", "aggregation", "fusion"]},
        "experiment": build_experiment_name(config), "status": "planned", "selection_reason": reason,
        "experiment_summary": "",
    }


def complete(row: dict[str, str], landscape: dict[str, dict]) -> None:
    result = landscape[row["experiment"]]
    row.update(status="completed", val_accuracy=f"{result['val_accuracy']:.6f}", val_f1=f"{result['val_f1']:.6f}")


def propose_random(rows, method, seed, budget, **_):
    if not rows:
        propose_random.order = random_search.select_configurations(rows=[], number_to_select=budget, seed=seed)
    return propose_random.order[len(rows)], "Random selection."


def propose_bayesian(rows, method, seed, startup_trials, **_):
    _, config, n_completed = bayesian_search.suggest_configuration(
        rows=rows, seed=seed, metric="val_accuracy", startup_trials=startup_trials)
    return config, f"Optuna TPE suggestion based on {n_completed} completed trial(s)."


def propose_llm(rows, method, log, seed, budget, **_):
    variant = LLM_VARIANTS[method]
    mode = variant["mode"]
    prompt, config_by_id, available_ids = llm_search.build_prompt(
        rows, mode, shuffle_seed=seed if variant["shuffle"] else None, budget=budget if variant["budget"] else None,
        isolation_principles=variant["isolation"])
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        text = llm_search.query_llm(prompt)
        log.append({"type": "decision", "trial": len(rows) + 1, "attempt": attempt, "prompt": prompt,
                    "response": text})
        try:
            config_id, reason, _, _ = llm_search.parse_response(text, mode)
        except ValueError:
            continue
        if config_id in available_ids:
            return config_by_id[config_id], reason
    raise RuntimeError(f"{method}: no valid configuration after {MAX_LLM_ATTEMPTS} attempts")


def summarise_llm(rows, row, log):
    prompt = summarize_experiment.build_prompt(rows, row)
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        text = summarize_experiment.query_llm(prompt)
        log.append({"type": "summary", "trial": len(rows), "attempt": attempt, "prompt": prompt, "response": text})
        try:
            row["experiment_summary"] = summarize_experiment.parse_summary(text)
            return
        except ValueError:
            continue
    raise RuntimeError(f"No valid summary after {MAX_LLM_ATTEMPTS} attempts")


PROPOSERS = {"random": propose_random, "bayesian": propose_bayesian, **{name: propose_llm for name in LLM_VARIANTS}}


def replay(method, replicate, landscape, budget, startup_trials):
    seed = 1000 + replicate
    rows, log = [], []
    for number in range(1, budget + 1):
        config, reason = PROPOSERS[method](rows=rows, method=method, seed=seed, budget=budget,
                                           startup_trials=startup_trials, log=log)
        row = base_row(config, method, number, reason)
        complete(row, landscape)
        rows.append(row)
        if method == "llm_summary":
            summarise_llm(rows, row, log)
    if log:
        log_dir = REPLAY_DIR / "llm_logs" / method
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"replicate_{replicate:03d}.json").write_text(json.dumps(log, indent=1))
    return [{"method": method, "replicate": replicate, "seed": seed, "trial": index, "experiment": row["experiment"],
             "val_accuracy": float(row["val_accuracy"]), "val_f1": float(row["val_f1"]),
             "selection_reason": row["selection_reason"], "experiment_summary": row["experiment_summary"]}
            for index, row in enumerate(rows, start=1)]


def run_method(method, replicates, landscape_name, budget, startup_trials):
    landscape = load_landscape(landscape_name)
    path = REPLAY_DIR / f"{method}.csv"
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    trials = []
    if path.exists():
        saved = pd.read_csv(path)
        saved = saved.loc[saved["landscape"].eq(landscape_name) & saved["replicate"].lt(replicates)]
        # Keep only replicates that ran to the full budget
        complete_ids = saved.groupby("replicate")["trial"].max().loc[lambda s: s.eq(budget)].index
        trials = saved.loc[saved["replicate"].isin(complete_ids)].drop(columns="landscape").to_dict("records")
    done = {row["replicate"] for row in trials}
    if done:
        print(f"{method}: resuming, {len(done)} replicate(s) already saved")
    for replicate in range(replicates):
        if replicate in done:
            continue
        trials += replay(method, replicate, landscape, budget, startup_trials)
        # Written after every replicate so an interrupted LLM run keeps finished replicates
        pd.DataFrame(trials).assign(landscape=landscape_name).to_csv(path, index=False)
        print(f"{method}: replicate {replicate + 1}/{replicates} done")


def bootstrap_ci(values: np.ndarray, rng) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    means = rng.choice(values, size=(N_BOOTSTRAP, len(values))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def analyze():
    frames = [pd.read_csv(path) for path in sorted(REPLAY_DIR.glob("*.csv"))]
    if not frames:
        raise RuntimeError(f"No replay files in {REPLAY_DIR}")
    trials = pd.concat(frames, ignore_index=True)
    full = pd.read_csv(LANDSCAPES["full"]).set_index("experiment")
    trials["full_val_accuracy"] = trials["experiment"].map(full["val_accuracy"])
    trials["runtime_hours"] = trials["experiment"].map(full["runtime_hours"])
    ranked = full["val_accuracy"].sort_values(ascending=False)
    optimum, top3 = ranked.iloc[0], set(ranked.index[:3])

    trials = trials.sort_values(["method", "replicate", "trial"])
    group = trials.groupby(["method", "replicate"])
    trials["best_so_far"] = group["full_val_accuracy"].cummax()
    trials["regret"] = optimum - trials["best_so_far"]
    trials["found_best"] = group["experiment"].transform(lambda s: (s == ranked.index[0]).cummax())
    trials["found_top3"] = group["experiment"].transform(lambda s: s.isin(top3).cummax())
    trials["cumulative_gpu_hours"] = group["runtime_hours"].cumsum()

    rng = np.random.default_rng(0)
    rows = []
    for (method, trial), sub in trials.groupby(["method", "trial"]):
        low, high = bootstrap_ci(sub["best_so_far"].to_numpy(), rng)
        rows.append({"method": method, "trial": trial, "n_replicates": len(sub),
                     "mean_best_val_accuracy": sub["best_so_far"].mean(), "sd_best_val_accuracy": sub["best_so_far"].std(),
                     "ci95_low": low, "ci95_high": high, "mean_regret": sub["regret"].mean(),
                     "p_found_best": sub["found_best"].mean(), "p_found_top3": sub["found_top3"].mean(),
                     "mean_cumulative_gpu_hours": sub["cumulative_gpu_hours"].mean()})
    curves = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUTPUT_DIR / "replay_curves.csv", index=False)
    trials.to_csv(OUTPUT_DIR / "replay_trials_scored.csv", index=False)
    print(curves.loc[curves["trial"].isin([3, 5, 10])].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    trials["model_setting"] = trials["experiment"].str.split("/").str[0]
    first = trials.loc[trials["trial"].eq(1)]
    mix = pd.crosstab(trials["method"], trials["model_setting"], normalize="index").add_prefix("fraction_")
    mix["n_distinct_first_picks"] = first.groupby("method")["experiment"].nunique()
    mix["first_multimodal_trial_mean"] = trials.loc[trials["model_setting"].eq("multimodal")].groupby(
        ["method", "replicate"])["trial"].min().groupby("method").mean()
    mix["replicates_without_multimodal"] = trials.groupby("method")["replicate"].nunique() - trials.loc[
        trials["model_setting"].eq("multimodal")].groupby("method")["replicate"].nunique().reindex(mix.index).fillna(0)
    mix = mix.reindex([m for m in METHOD_ORDER if m in mix.index])
    mix.to_csv(OUTPUT_DIR / "replay_selection_mix.csv")
    print(mix.round(2).to_string())

    present = set(curves["method"])
    plot_curves(curves, optimum, [m for m in ["random", "bayesian", "llm_metrics", "llm_summary"] if m in present],
                OUTPUT_DIR / "replay_best_validation.png")
    ablation = [m for m in ["random", *[v for v in LLM_VARIANTS if v != "llm_summary"]] if m in present]
    if len(ablation) > 2:
        plot_curves(curves, optimum, ablation, OUTPUT_DIR / "replay_llm_prompt_ablation.png")
    print(f"Saved {OUTPUT_DIR}")


def plot_curves(curves, optimum, methods, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for method in methods:
        sub = curves.loc[curves["method"].eq(method)]
        label = f"{method} (n = {sub['n_replicates'].max()})"
        axes[0].plot(sub["trial"], sub["mean_best_val_accuracy"], color=COLOURS[method], marker="o", ms=3, label=label)
        axes[0].fill_between(sub["trial"], sub["ci95_low"], sub["ci95_high"], color=COLOURS[method], alpha=0.2)
        axes[1].plot(sub["trial"], sub["p_found_top3"], color=COLOURS[method], marker="o", ms=3, label=label)
        axes[2].plot(sub["mean_cumulative_gpu_hours"], sub["mean_best_val_accuracy"], color=COLOURS[method],
                     marker="o", ms=3, label=label)
    axes[0].axhline(optimum, color="black", ls="--", lw=1, label=f"Landscape optimum ({optimum:.3f})")
    axes[2].axhline(optimum, color="black", ls="--", lw=1)
    axes[0].set(xlabel="Experiment number", ylabel="Best validation accuracy so far (full training)",
                title="Best found (mean, 95% bootstrap CI)")
    axes[1].set(xlabel="Experiment number", ylabel="Fraction of replicates", ylim=(0, 1.02),
                title="Found one of the top-3 configurations")
    axes[2].set(xlabel="Cumulative full-training GPU hours", ylabel="Best validation accuracy so far",
                title="Best found vs compute (mean)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=["random", "bayesian"])
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--landscape", choices=sorted(LANDSCAPES), default="full",
                        help="Validation scores the methods see as feedback (results are always scored on full).")
    parser.add_argument("--startup-trials", type=int, default=3, help="Same default as bayesian_search.py.")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if not args.analyze_only:
        for method in args.methods:
            run_method(method, args.replicates, args.landscape, args.budget, args.startup_trials)
    analyze()


if __name__ == "__main__":
    main()
