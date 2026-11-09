"""Replay the search methods many times on the completed 51-configuration landscape (lookup instead of training).

Each replicate starts from an empty history; when a method picks a configuration, its validation accuracy/F1 are
looked up from a landscape results file instead of being trained. Original live selectors are preserved alongside
the additional classical baselines. LLM methods need OpenAI credentials.

    python replay_search.py --methods random bayesian --replicates 200
    python replay_search.py --methods llm_metrics llm_summary --replicates 5
    python replay_search.py --methods llm_metrics --replicates 10 --llm-model anthropic_claude_opus_5_5
    python replay_search.py --analyze-only

Replays with a non-default --llm-model are written to results/lookup_replays/models/<model>/ and
analysis/lookup_replay/models/<model>/; analysis/lookup_replay/model_comparison.csv compares all models.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

import bayesian_search
import baseline_io
import classical_search
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
BASE_REPLAY_DIR, BASE_OUTPUT_DIR = REPLAY_DIR, OUTPUT_DIR
DEFAULT_LLM_MODEL = llm_search.LLM_MODEL
# Prompt variants: shuffle = random candidate order and IDs per replicate; budget = state experiment k of N and
# make best-found accuracy the only objective; isolation = keep the one-factor-at-a-time / complexity principles;
# gpu = show measured GPU hours of completed runs and ask to keep total GPU hours low
LLM_VARIANTS = {
    "llm_metrics": dict(mode="metrics", shuffle=False, budget=False, isolation=True, gpu=False),
    "llm_summary": dict(mode="summary", shuffle=False, budget=False, isolation=True, gpu=False),
    "llm_metrics_shuffled": dict(mode="metrics", shuffle=True, budget=False, isolation=True, gpu=False),
    "llm_metrics_budget": dict(mode="metrics", shuffle=False, budget=True, isolation=True, gpu=False),
    "llm_metrics_no_isolation": dict(mode="metrics", shuffle=False, budget=False, isolation=False, gpu=False),
    "llm_metrics_combined": dict(mode="metrics", shuffle=True, budget=True, isolation=False, gpu=False),
    "llm_metrics_combined_gpu": dict(mode="metrics", shuffle=True, budget=True, isolation=False, gpu=True),
}
METHOD_ORDER = [*classical_search.METHODS, *LLM_VARIANTS]
COLOURS = {"random": "#7f7f7f", "bayesian": "#1f77b4", "llm_metrics": "#d62728", "llm_summary": "#2ca02c",
           "llm_metrics_shuffled": "#ff7f0e", "llm_metrics_budget": "#9467bd",
           "llm_metrics_no_isolation": "#8c564b", "llm_metrics_combined": "#e377c2",
           "llm_metrics_combined_gpu": "#17becf"}
MAX_LLM_ATTEMPTS = 3
N_BOOTSTRAP = 2000


def load_landscape(name: str) -> dict[str, dict]:
    table = pd.read_csv(LANDSCAPES[name])
    table = table.loc[table["status"].eq("completed")]
    if "runtime_hours" not in table:
        table["runtime_hours"] = np.nan
    expected = {build_experiment_name(config) for config in generate_all_configurations(runnable_only=True)}
    if len(expected) != 51 or len(table) != 51 or not table["experiment"].is_unique or set(table["experiment"]) != expected:
        raise RuntimeError(f"{name} landscape must contain exactly the 51 unique completed configurations.")
    for metric in ("val_accuracy", "val_f1"):
        if not np.isfinite(table[metric]).all() or not table[metric].between(0, 1).all():
            raise ValueError(f"Invalid landscape {metric}.")
    if name == "full" and (not np.isfinite(table["runtime_hours"]).all() or (table["runtime_hours"] <= 0).any()):
        raise ValueError("Full landscape requires finite positive training costs.")
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
    row.update(status="completed", val_accuracy=f"{result['val_accuracy']:.6f}", val_f1=f"{result['val_f1']:.6f}",
               runtime_hours=f"{result['runtime_hours']:.4f}")


def propose_random(rows, method, seed, budget, state, **_):
    if not rows:
        state["order"] = random_search.select_configurations(rows=[], number_to_select=budget, seed=seed)
    return state["order"][len(rows)], "Random selection."


def propose_bayesian(rows, method, seed, startup_trials, **_):
    _, config, n_completed = bayesian_search.suggest_configuration(
        rows=rows, seed=seed, metric="val_accuracy", startup_trials=startup_trials)
    return config, f"Optuna TPE suggestion based on {n_completed} completed trial(s)."


def propose_llm(rows, method, log, seed, budget, **_):
    variant = LLM_VARIANTS[method]
    mode = variant["mode"]
    prompt, config_by_id, available_ids = llm_search.build_prompt(
        rows, mode, shuffle_seed=seed if variant["shuffle"] else None, budget=budget if variant["budget"] else None,
        isolation_principles=variant["isolation"], gpu_hours_aware=variant["gpu"])
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


def propose_classical(rows, method, seed, startup_trials, state, **_):
    # Explicit allowlist at the optimiser boundary; no test, cost, summaries, or ranks.
    visible = [{key: row[key] for key in (*classical_search.FIELDS, "status", "val_accuracy", "val_f1")}
               for row in rows]
    config, diagnostics = classical_search.suggest(visible, method, seed, startup_trials)
    state["diagnostics"] = diagnostics
    return config, f"{method}: {json.dumps(diagnostics, sort_keys=True)}"


PROPOSERS = {"random": propose_random, "bayesian": propose_bayesian,
             **{name: propose_classical for name in classical_search.METHODS if name not in {"random", "bayesian"}},
             **{name: propose_llm for name in LLM_VARIANTS}}


def replay(method, replicate, landscape, budget, startup_trials, base_seed=1000):
    seed = base_seed + replicate
    rows, log, state = [], [], {}
    eligible = {classical_search.key(c) for c in classical_search.catalogue(method)}
    if not 1 <= budget <= len(eligible):
        raise ValueError(f"{method}: budget must be between 1 and {len(eligible)}.")
    used = set()
    for number in range(1, budget + 1):
        config, reason = PROPOSERS[method](rows=rows, method=method, seed=seed, budget=budget,
                                           startup_trials=startup_trials, log=log, state=state)
        config_key = classical_search.key(classical_search.configuration(config))
        if config_key not in eligible or config_key in used:
            raise ValueError("Proposer selected an invalid or repeated configuration.")
        used.add(config_key)
        row = base_row(config, method, number, reason)
        row.update(state.get("diagnostics", {}))
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
             "selection_reason": row["selection_reason"], "experiment_summary": row["experiment_summary"],
             **{name: row[name] for name in ("phase", "proposal_attempts", "fallback", "gp_jitter") if name in row}}
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
    if REPLAY_DIR != BASE_REPLAY_DIR:
        # Non-LLM reference methods are model-independent, so reuse the default replays
        frames += [pd.read_csv(BASE_REPLAY_DIR / f"{m}.csv") for m in ("random", "bayesian")
                   if (BASE_REPLAY_DIR / f"{m}.csv").exists()]
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


def compare_models():
    """Final (trial 10) metrics per LLM variant for the default model and every model under models/."""
    sources = {DEFAULT_LLM_MODEL: BASE_OUTPUT_DIR / "replay_curves.csv"}
    sources.update({path.parent.name: path for path in sorted((BASE_OUTPUT_DIR / "models").glob("*/replay_curves.csv"))})
    tables = []
    for model, path in sources.items():
        if not path.exists():
            continue
        curves = pd.read_csv(path)
        curves = curves.loc[curves["method"].isin(LLM_VARIANTS)]
        tables.append(curves.loc[curves["trial"].eq(curves["trial"].max())].assign(model=model))
    if len(tables) < 2:
        return
    table = pd.concat(tables, ignore_index=True)
    columns = ["method", "model", "n_replicates", "mean_best_val_accuracy", "ci95_low", "ci95_high",
               "p_found_best", "p_found_top3", "mean_cumulative_gpu_hours"]
    table = table[columns].sort_values(["method", "model"])
    table.to_csv(BASE_OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison (after the full budget):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    methods = [m for m in LLM_VARIANTS if m in set(table["method"])]
    models = list(dict.fromkeys(table["model"]))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for i, model in enumerate(models):
        sub = table.loc[table["model"].eq(model)].set_index("method").reindex(methods)
        x = np.arange(len(methods)) + (i - (len(models) - 1) / 2) * width
        error = np.vstack([sub["mean_best_val_accuracy"] - sub["ci95_low"], sub["ci95_high"] - sub["mean_best_val_accuracy"]])
        axes[0].bar(x, sub["mean_best_val_accuracy"], width, yerr=np.nan_to_num(error), capsize=3, label=model)
        axes[1].bar(x, sub["p_found_best"], width, label=model)
    for ax, ylabel in zip(axes, ["Best validation accuracy after budget (95% CI)", "Fraction of replicates finding the optimum"]):
        ax.set_xticks(range(len(methods)), [m.removeprefix("llm_") for m in methods], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylim(0.6, 0.74)
    axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(BASE_OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)


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


def baseline_settings(args):
    return baseline_io.protocol(SEARCH_DIR, LANDSCAPES[args.landscape], args.landscape, args.budget,
                                args.startup_trials, args.seed,
                                functions=(replay, base_row, complete, propose_random, propose_bayesian, propose_classical))


def analyze_baselines(args, directory, metadata):
    import baseline_analysis

    frames, sources = [], []
    for method in args.methods:
        path = directory / f"{method}.csv" if method in classical_search.METHODS else REPLAY_DIR / f"{method}.csv"
        if not path.exists():
            print(f"No matching saved results: {path}")
            continue
        table = baseline_io.validate_trials(pd.read_csv(path, keep_default_na=False), method, args.budget,
                                           args.landscape, args.seed if method in classical_search.METHODS else None)
        table = table.loc[table["replicate"].lt(args.replicates)].copy()
        if table.empty:
            continue
        model = "classical" if method in classical_search.METHODS else args.llm_model
        table["model"] = model
        table["series"] = method if model == "classical" else f"{model}/{method}"
        frames.append(table)
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    output = BASE_OUTPUT_DIR / "baselines" / directory.relative_to(BASE_REPLAY_DIR / "baselines")
    if any(method in LLM_VARIANTS for method in args.methods):
        output = output / "comparisons" / args.llm_model
    full = pd.read_csv(LANDSCAPES["full"])
    load_landscape("full")  # Validate the offline reference before producing figures.
    baseline_analysis.analyze(frames, full, output, dict(metadata, analysis_sources=sources,
                                                       full_landscape_sha256=hashlib.sha256(LANDSCAPES["full"].read_bytes()).hexdigest(),
                                                       requested_replicates=args.replicates,
                                                       llm_provenance="legacy files; serving settings not recorded"))


def build_parser():
    import llm_client

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=["random", "bayesian"])
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--landscape", choices=sorted(LANDSCAPES), default="full",
                        help="Validation scores the methods see as feedback (results are always scored on full).")
    parser.add_argument("--startup-trials", type=int, default=3, help="Same default as bayesian_search.py.")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Gateway model id for the LLM methods.")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Responses defaults: 500/400 for default model, 2000 otherwise. Local Chat default: 16384.")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed; new Chat runs also derive explicit per-request seeds.")
    llm_client.add_arguments(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.methods = list(dict.fromkeys(args.methods))
    if args.replicates < 1 or args.startup_trials < 1 or args.seed < 0 or args.seed + args.replicates > 2**32:
        parser.error("Use positive replicates/startup counts and seeds in the uint32 range.")
    for method in args.methods:
        limit = len(classical_search.catalogue(method))
        if not 1 <= args.budget <= limit:
            parser.error(f"{method} supports budgets from 1 to {limit}.")
    if args.analysis_inputs:
        if not args.analyze_only:
            parser.error("--analysis-inputs requires --analyze-only.")
        import llm_analysis
        llm_analysis.run(args, sys.modules[__name__])
        return
    if args.llm_api == "chat":
        if args.analyze_only:
            parser.error("Use --analysis-inputs with explicit protocol directories for offline Chat analysis.")
        import llm_replay_io
        llm_replay_io.run(args, sys.modules[__name__])
        return
    if args.smoke_test or args.condition is not None or args.thinking is not None or args.reasoning_effort is not None or args.model_revision is not None or args.context_window is not None or args.decoding_profile != "controlled" or any(
            value is not None for value in (args.temperature, args.top_p, args.top_k, args.presence_penalty)):
        parser.error("Local inference controls require --llm-api chat; the original Responses path is unchanged.")
    if not args.analyze_only and any(method in LLM_VARIANTS for method in args.methods):
        from llm_client import MODELS
        if args.llm_model in MODELS:
            parser.error("New local-model runs require --llm-api chat. Historical local CSVs are read-only.")
    if args.seed != 1000 and any(method in LLM_VARIANTS for method in args.methods) and not args.analyze_only:
        parser.error("--seed applies to classical generation; existing LLM seed behaviour is unchanged.")

    global REPLAY_DIR, OUTPUT_DIR
    llm_search.LLM_MODEL = summarize_experiment.LLM_MODEL = args.llm_model
    # Other models write longer reasons; the default limits truncate their JSON
    tokens = args.max_output_tokens or (None if args.llm_model == DEFAULT_LLM_MODEL else 2000)
    if tokens:
        llm_search.LLM_MAX_OUTPUT_TOKENS = summarize_experiment.SUMMARY_MAX_OUTPUT_TOKENS = tokens
    if args.llm_model != DEFAULT_LLM_MODEL:
        REPLAY_DIR = BASE_REPLAY_DIR / "models" / args.llm_model
        OUTPUT_DIR = BASE_OUTPUT_DIR / "models" / args.llm_model
        if not set(args.methods) <= set(LLM_VARIANTS) and not args.analyze_only:
            parser.error("--llm-model only applies to LLM methods; random/bayesian replays are shared")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if "tpe_discrete" in args.methods and not args.analyze_only:
        from tpe_acquisition import require_compatible_version
        require_compatible_version()
    classical = [method for method in args.methods if method in classical_search.METHODS]
    if classical:
        relative, metadata = baseline_settings(args)
        directory = BASE_REPLAY_DIR / "baselines" / relative
        if args.analyze_only:
            # Never merge implementation versions. Each matching directory gets its own report.
            pattern = f"budget_{args.budget:03d}_startup_{args.startup_trials:03d}_seed_{args.seed}_p*"
            matches = sorted((BASE_REPLAY_DIR / "baselines" / args.landscape).glob(pattern))
            if not matches:
                if set(args.methods) <= {"random", "bayesian"} and (args.budget, args.startup_trials, args.seed, args.landscape) == (10, 3, 1000, "full"):
                    # Archive compatibility: read historical CSVs, write analysis to a new location.
                    OUTPUT_DIR = BASE_OUTPUT_DIR / "baselines" / "legacy"
                    print("Analysing archived replays; package/startup provenance is unverified.")
                    analyze()
                    return
                raise FileNotFoundError("No baseline outputs match these settings.")
            for match in matches:
                stored = json.loads((match / "metadata.json").read_text())
                expected = (args.budget, args.startup_trials, args.seed, args.landscape)
                if tuple(stored[k] for k in ("budget", "startup_trials", "base_seed", "landscape")) != expected:
                    raise ValueError("Directory name and stored settings disagree.")
                if stored["landscape_sha256"] != metadata["landscape_sha256"]:
                    raise ValueError("Landscape has changed since this replay; refusing to rescore it.")
                analyze_baselines(args, match, stored)
            return
        print(f"Classical results: {directory}")
        feedback = load_landscape(args.landscape)
        for method in classical:
            baseline_io.run_method(directory, metadata, method, args.replicates, feedback, replay)
        for method in args.methods:
            if method in LLM_VARIANTS:
                run_method(method, args.replicates, args.landscape, args.budget, args.startup_trials)
        analyze_baselines(args, directory, metadata)
        return
    if not args.analyze_only:
        for method in args.methods:
            run_method(method, args.replicates, args.landscape, args.budget, args.startup_trials)
    analyze()
    compare_models()


if __name__ == "__main__":
    main()
