import csv
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


SEARCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SEARCH_DIR / "results"
OUTPUT_DIR = SEARCH_DIR / "analysis" / "landscape"

SHARD_FILES = [
    RESULTS_DIR / "proxy_landscape_gpu1.csv",
    RESULTS_DIR / "proxy_landscape_gpu2.csv",
]

FIELDNAMES = [
    "experiment_id",
    "search_method",
    "model_setting",
    "profile_setting",
    "rna_encoder",
    "image_encoder",
    "aggregation",
    "fusion",
    "experiment",
    "status",
    "val_accuracy",
    "val_f1",
    "test_accuracy",
    "test_f1",
    "selection_reason",
    "experiment_summary",
    "started_at",
    "finished_at",
    "return_code",
    "run_directory",
    "log_file",
    "error_message",
]

ARCHITECTURE_FIELDS = [
    "model_setting",
    "profile_setting",
    "rna_encoder",
    "image_encoder",
    "aggregation",
    "fusion",
]

BUDGETS = [3, 5, 8, 10, 12, 15, 20]
MONTE_CARLO_REPEATS = 10000
RANDOM_SEED = 12345


def read_rows():
    rows = []

    for path in SHARD_FILES:
        if not path.exists():
            continue

        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    if not rows:
        fallback = RESULTS_DIR / "proxy_landscape_results.csv"

        if fallback.exists():
            with fallback.open(newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))

    # Deduplicate by experiment_id, preferring completed rows.
    by_id = {}

    status_priority = {
        "completed": 4,
        "running": 3,
        "failed": 2,
        "planned": 1,
    }

    for row in rows:
        experiment_id = row["experiment_id"]

        current = by_id.get(experiment_id)

        if current is None:
            by_id[experiment_id] = row
            continue

        if status_priority.get(row["status"], 0) > status_priority.get(
            current["status"], 0
        ):
            by_id[experiment_id] = row

    return sorted(
        by_id.values(),
        key=lambda row: int(row["experiment_id"].split("_")[-1]),
    )


def parse_float(value):
    if value is None or value == "":
        return None

    return float(value)


def runtime_minutes(row):
    start = row.get("started_at")
    end = row.get("finished_at")

    if not start or not end:
        return None

    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        return None

    return (end_dt - start_dt).total_seconds() / 60.0


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def probability_hit_at_least_one(n, k, budget):
    """
    Exact probability of sampling >=1 member of a target set of size k
    in budget draws without replacement.
    """
    if k <= 0:
        return 0.0

    if budget >= n:
        return 1.0

    if n - k < budget:
        return 1.0

    miss = math.comb(n - k, budget) / math.comb(n, budget)

    return 1.0 - miss


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_rows()

    completed = []

    for row in rows:
        if row["status"] != "completed":
            continue

        acc = parse_float(row["val_accuracy"])

        if acc is None:
            continue

        item = dict(row)
        item["val_accuracy"] = acc
        item["val_f1"] = parse_float(row["val_f1"])
        item["runtime_minutes"] = runtime_minutes(row)

        completed.append(item)

    statuses = defaultdict(int)

    for row in rows:
        statuses[row["status"]] += 1

    print("\n===== LANDSCAPE STATUS =====")
    print("Total configurations:", len(rows))
    print("Completed:", statuses["completed"])
    print("Running:", statuses["running"])
    print("Planned:", statuses["planned"])
    print("Failed:", statuses["failed"])

    if not completed:
        print("\nNo completed configurations yet.")
        return

    completed.sort(
        key=lambda row: row["val_accuracy"],
        reverse=True,
    )

    best = completed[0]
    best_acc = best["val_accuracy"]

    print("\n===== CURRENT BEST =====")
    print("Experiment:", best["experiment_id"])
    print("Configuration:", best["experiment"])
    print(f"Validation accuracy: {best_acc:.4f}")
    if best["val_f1"] is not None:
        print(f"Validation F1: {best['val_f1']:.4f}")

    # ---------------------------------------------------------
    # Top configurations
    # ---------------------------------------------------------

    top_rows = []

    for rank, row in enumerate(completed, start=1):
        item = {
            "rank": rank,
            "experiment_id": row["experiment_id"],
            "model_setting": row["model_setting"],
            "profile_setting": row["profile_setting"],
            "rna_encoder": row["rna_encoder"],
            "image_encoder": row["image_encoder"],
            "aggregation": row["aggregation"],
            "fusion": row["fusion"],
            "val_accuracy": row["val_accuracy"],
            "val_f1": row["val_f1"],
            "runtime_minutes": row["runtime_minutes"],
            "experiment": row["experiment"],
        }

        top_rows.append(item)

    write_csv(
        OUTPUT_DIR / "top_configurations.csv",
        [
            "rank",
            "experiment_id",
            "model_setting",
            "profile_setting",
            "rna_encoder",
            "image_encoder",
            "aggregation",
            "fusion",
            "val_accuracy",
            "val_f1",
            "runtime_minutes",
            "experiment",
        ],
        top_rows,
    )

    # ---------------------------------------------------------
    # Architecture effects
    # ---------------------------------------------------------

    architecture_rows = []

    for field in ARCHITECTURE_FIELDS:
        groups = defaultdict(list)

        for row in completed:
            value = row[field] or "none"
            groups[value].append(row["val_accuracy"])

        for value, scores in sorted(groups.items()):
            architecture_rows.append(
                {
                    "factor": field,
                    "value": value,
                    "n": len(scores),
                    "mean_val_accuracy": mean(scores),
                    "median_val_accuracy": median(scores),
                    "best_val_accuracy": max(scores),
                    "worst_val_accuracy": min(scores),
                }
            )

    write_csv(
        OUTPUT_DIR / "architecture_effects.csv",
        [
            "factor",
            "value",
            "n",
            "mean_val_accuracy",
            "median_val_accuracy",
            "best_val_accuracy",
            "worst_val_accuracy",
        ],
        architecture_rows,
    )

    # ---------------------------------------------------------
    # Runtime summary
    # ---------------------------------------------------------

    runtime_rows = [
        row
        for row in completed
        if row["runtime_minutes"] is not None
    ]

    runtime_summary = []

    if runtime_rows:
        runtime_values = [
            row["runtime_minutes"]
            for row in runtime_rows
        ]

        runtime_summary.append(
            {
                "group": "all",
                "n": len(runtime_values),
                "mean_minutes": mean(runtime_values),
                "median_minutes": median(runtime_values),
                "min_minutes": min(runtime_values),
                "max_minutes": max(runtime_values),
                "total_gpu_hours": sum(runtime_values) / 60,
            }
        )

        for model_setting in sorted(
            set(row["model_setting"] for row in runtime_rows)
        ):
            group = [
                row["runtime_minutes"]
                for row in runtime_rows
                if row["model_setting"] == model_setting
            ]

            runtime_summary.append(
                {
                    "group": f"model_setting={model_setting}",
                    "n": len(group),
                    "mean_minutes": mean(group),
                    "median_minutes": median(group),
                    "min_minutes": min(group),
                    "max_minutes": max(group),
                    "total_gpu_hours": sum(group) / 60,
                }
            )

    write_csv(
        OUTPUT_DIR / "runtime_summary.csv",
        [
            "group",
            "n",
            "mean_minutes",
            "median_minutes",
            "min_minutes",
            "max_minutes",
            "total_gpu_hours",
        ],
        runtime_summary,
    )

    # ---------------------------------------------------------
    # Search-budget analysis
    # Only scientifically final once all 51 are complete.
    # ---------------------------------------------------------

    if len(completed) == 51:
        scores = [
            row["val_accuracy"]
            for row in completed
        ]

        n = len(scores)

        sorted_scores = sorted(scores, reverse=True)

        top3_threshold = sorted_scores[2]

        top10_count = max(
            1,
            math.ceil(0.10 * n),
        )
        top10_threshold = sorted_scores[top10_count - 1]

        within_1pt_threshold = best_acc - 0.01
        within_95pct_threshold = best_acc * 0.95
        within_99pct_threshold = best_acc * 0.99

        target_sets = {
            "global_best": sum(
                score == best_acc
                for score in scores
            ),
            "top_3": sum(
                score >= top3_threshold
                for score in scores
            ),
            "top_10_percent": sum(
                score >= top10_threshold
                for score in scores
            ),
            "within_0.01_absolute": sum(
                score >= within_1pt_threshold
                for score in scores
            ),
            "within_95_percent_of_optimum": sum(
                score >= within_95pct_threshold
                for score in scores
            ),
            "within_99_percent_of_optimum": sum(
                score >= within_99pct_threshold
                for score in scores
            ),
        }

        random.seed(RANDOM_SEED)

        budget_rows = []

        for budget in BUDGETS:
            if budget > n:
                continue

            simulated_best = []

            for _ in range(MONTE_CARLO_REPEATS):
                sample = random.sample(
                    scores,
                    budget,
                )
                simulated_best.append(max(sample))

            row = {
                "budget": budget,
                "mean_best_accuracy": mean(simulated_best),
                "median_best_accuracy": median(simulated_best),
                "mean_regret": best_acc - mean(simulated_best),
            }

            for name, k in target_sets.items():
                row[f"p_{name}"] = probability_hit_at_least_one(
                    n=n,
                    k=k,
                    budget=budget,
                )

            budget_rows.append(row)

        budget_fields = [
            "budget",
            "mean_best_accuracy",
            "median_best_accuracy",
            "mean_regret",
        ] + [
            f"p_{name}"
            for name in target_sets
        ]

        write_csv(
            OUTPUT_DIR / "random_budget_analysis.csv",
            budget_fields,
            budget_rows,
        )

        print("\n===== RANDOM SEARCH TARGET SETS =====")

        for name, count in target_sets.items():
            print(f"{name}: {count}/51 configurations")

        print("\n===== RANDOM SEARCH BUDGETS =====")

        for row in budget_rows:
            print(
                f"Budget {row['budget']:2d}: "
                f"mean best={row['mean_best_accuracy']:.4f}, "
                f"P(top3)={row['p_top_3']:.3f}, "
                f"P(within 0.01)={row['p_within_0.01_absolute']:.3f}"
            )

    else:
        print(
            "\nBudget analysis skipped because the complete "
            "51-configuration landscape is not available yet."
        )

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------

    accuracies = [
        row["val_accuracy"]
        for row in completed
    ]

    plt.figure(figsize=(7, 5))
    plt.hist(accuracies, bins=min(12, max(5, len(accuracies))))
    plt.xlabel("Proxy validation accuracy")
    plt.ylabel("Number of configurations")
    plt.title("Configuration Performance Landscape")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "validation_accuracy_distribution.png",
        dpi=200,
    )
    plt.close()

    if runtime_rows:
        x = [
            row["runtime_minutes"]
            for row in runtime_rows
        ]
        y = [
            row["val_accuracy"]
            for row in runtime_rows
        ]

        plt.figure(figsize=(7, 5))
        plt.scatter(x, y)
        plt.xlabel("Proxy runtime (minutes)")
        plt.ylabel("Proxy validation accuracy")
        plt.title("Performance vs Experiment Cost")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "accuracy_vs_runtime.png",
            dpi=200,
        )
        plt.close()

    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
