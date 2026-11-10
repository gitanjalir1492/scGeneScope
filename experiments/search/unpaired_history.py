"""Unpaired-history control: does the LLM actually use the experiment history?

Per replicate, the (val_accuracy, val_f1) pairs of the 51 landscape configurations are randomly permuted across
configurations, so every metric the LLM sees belongs to a different experiment. The feedback stays internally
consistent within a replicate, but carries no information about which configuration is good. Runtime (GPU hours)
stays with its own configuration. Selections are still scored on the true full landscape.

If performance stays at the paired level, the LLM relies on priors rather than the history; if it drops towards
random search, it is using the history.

    python unpaired_history.py --llm-model openai_gpt6_astra --methods llm_metrics --replicates 50
    python unpaired_history.py --llm-model openai_gpt6_astra --analyze-only

Outputs: results/lookup_replays/unpaired_history/<model>/ and analysis/lookup_replay/unpaired_history/<model>/;
analysis/lookup_replay/unpaired_history/paired_vs_unpaired.csv compares against the normal (paired) replays.
"""

import argparse

import numpy as np
import pandas as pd

import llm_search
import replay_search as rs
import summarize_experiment

COMPARISON_DIR = rs.BASE_OUTPUT_DIR / "unpaired_history"
_replay = rs.replay


def unpaired_landscape(landscape: dict[str, dict], seed: int) -> dict[str, dict]:
    names = sorted(landscape)
    donors = np.random.default_rng([seed, 51]).permutation(names)
    return {name: {**landscape[name], "val_accuracy": landscape[donor]["val_accuracy"],
                   "val_f1": landscape[donor]["val_f1"]} for name, donor in zip(names, donors)}


def unpaired_replay(method, replicate, landscape, budget, startup_trials, base_seed=1000):
    return _replay(method, replicate, unpaired_landscape(landscape, base_seed + replicate), budget, startup_trials,
                   base_seed)


def compare():
    rows = []
    for path in sorted(COMPARISON_DIR.glob("*/replay_curves.csv")):
        model = path.parent.name
        paired = rs.BASE_OUTPUT_DIR / ("replay_curves.csv" if model == rs.DEFAULT_LLM_MODEL else f"models/{model}/replay_curves.csv")
        for history, source in (("unpaired", path), ("paired", paired)):
            if not source.exists():
                continue
            curves = pd.read_csv(source)
            unpaired_methods = set(pd.read_csv(path)["method"])
            curves = curves.loc[curves["method"].isin(unpaired_methods)]
            rows.append(curves.loc[curves["trial"].eq(curves["trial"].max())].assign(model=model, history=history))
    if not rows:
        return
    table = pd.concat(rows, ignore_index=True)
    table.loc[table["method"].isin(["random", "bayesian"]), ["model", "history"]] = ["-", "-"]
    table = table.drop_duplicates(["method", "model", "history"])
    columns = ["method", "model", "history", "n_replicates", "mean_best_val_accuracy", "ci95_low", "ci95_high",
               "p_found_best", "p_found_top3"]
    table = table[columns].sort_values(["method", "model", "history"])
    table.to_csv(COMPARISON_DIR / "paired_vs_unpaired.csv", index=False)
    print("\nPaired vs unpaired history (after the full budget):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--methods", nargs="+", choices=list(rs.LLM_VARIANTS), default=["llm_metrics"])
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    llm_search.LLM_MODEL = summarize_experiment.LLM_MODEL = args.llm_model
    tokens = args.max_output_tokens or (None if args.llm_model == rs.DEFAULT_LLM_MODEL else 2000)
    if tokens:
        llm_search.LLM_MAX_OUTPUT_TOKENS = summarize_experiment.SUMMARY_MAX_OUTPUT_TOKENS = tokens
    rs.REPLAY_DIR = rs.BASE_REPLAY_DIR / "unpaired_history" / args.llm_model
    rs.OUTPUT_DIR = COMPARISON_DIR / args.llm_model
    rs.replay = unpaired_replay

    if not args.analyze_only:
        for method in args.methods:
            rs.run_method(method, args.replicates, "full", args.budget, startup_trials=3)
    rs.analyze()
    compare()


if __name__ == "__main__":
    main()
