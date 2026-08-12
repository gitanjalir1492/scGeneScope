import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


SEARCH_DIR = Path(__file__).resolve().parent

RESULTS_PATH = (
    SEARCH_DIR
    / "results"
    / "proxy_search_results.csv"
)

OUTPUT_DIR = SEARCH_DIR / "analysis"

METHOD_ORDER = [
    "random",
    "bayesian",
    "llm_metrics",
    "llm_summary",
]


def read_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def trial_number(experiment_id):
    try:
        return int(
            experiment_id.split("_")[-1]
        )
    except (ValueError, IndexError):
        return 10**9


def runtime_minutes(row):
    started = row.get("started_at", "").strip()
    finished = row.get("finished_at", "").strip()

    if not started or not finished:
        return None

    try:
        start_time = datetime.fromisoformat(started)
        finish_time = datetime.fromisoformat(finished)
    except ValueError:
        return None

    return (
        finish_time - start_time
    ).total_seconds() / 60.0


def completed_rows(rows, method):
    method_rows = [
        row
        for row in rows
        if (
            row.get("search_method") == method
            and row.get("status") == "completed"
            and to_float(
                row.get("val_accuracy")
            ) is not None
        )
    ]

    return sorted(
        method_rows,
        key=lambda row:
            trial_number(
                row.get("experiment_id", "")
            ),
    )


def build_curves(rows):
    curve_rows = []
    summaries = []

    for method in METHOD_ORDER:
        completed = completed_rows(
            rows,
            method,
        )

        if not completed:
            continue

        best_accuracy = None
        best_f1 = None
        best_experiment = None
        best_config = None

        cumulative_minutes = 0.0

        for index, row in enumerate(
            completed,
            start=1,
        ):
            accuracy = to_float(
                row.get("val_accuracy")
            )

            f1 = to_float(
                row.get("val_f1")
            )

            runtime = runtime_minutes(row)

            if runtime is not None:
                cumulative_minutes += runtime

            if (
                best_accuracy is None
                or accuracy > best_accuracy
            ):
                best_accuracy = accuracy
                best_f1 = f1
                best_experiment = row.get(
                    "experiment_id",
                    "",
                )

                best_config = row.get(
                    "experiment",
                    "",
                )

            curve_rows.append(
                {
                    "search_method": method,
                    "trial": index,
                    "experiment_id":
                        row.get(
                            "experiment_id",
                            "",
                        ),
                    "val_accuracy":
                        accuracy,
                    "val_f1":
                        (
                            ""
                            if f1 is None
                            else f1
                        ),
                    "best_so_far_accuracy":
                        best_accuracy,
                    "cumulative_runtime_minutes":
                        cumulative_minutes,
                    "cumulative_runtime_hours":
                        cumulative_minutes / 60.0,
                }
            )

        summaries.append(
            {
                "search_method": method,
                "completed_trials":
                    len(completed),
                "best_experiment":
                    best_experiment,
                "best_val_accuracy":
                    best_accuracy,
                "best_val_f1":
                    (
                        ""
                        if best_f1 is None
                        else best_f1
                    ),
                "best_configuration":
                    best_config,
                "total_runtime_minutes":
                    cumulative_minutes,
                "total_runtime_hours":
                    cumulative_minutes / 60.0,
            }
        )

    return curve_rows, summaries


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries):
    print(
        "\nSearch comparison summary\n"
    )

    header = (
        f"{'Method':<16}"
        f"{'Trials':>8}"
        f"{'Best Acc':>12}"
        f"{'Best F1':>12}"
        f"{'Hours':>10}"
        f"  Best Experiment"
    )

    print(header)
    print("-" * len(header))

    for row in summaries:
        f1 = row["best_val_f1"]

        if f1 == "":
            f1_text = ""
        else:
            f1_text = f"{f1:.4f}"

        print(
            f"{row['search_method']:<16}"
            f"{row['completed_trials']:>8}"
            f"{row['best_val_accuracy']:>12.4f}"
            f"{f1_text:>12}"
            f"{row['total_runtime_hours']:>10.2f}"
            f"  {row['best_experiment']}"
        )


def plot_trials(curve_rows):
    plt.figure(
        figsize=(8, 5)
    )

    methods_present = []

    for method in METHOD_ORDER:
        points = [
            row
            for row in curve_rows
            if row["search_method"] == method
        ]

        if not points:
            continue

        methods_present.append(method)

        x = [
            row["trial"]
            for row in points
        ]

        y = [
            row["best_so_far_accuracy"]
            for row in points
        ]

        plt.plot(
            x,
            y,
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Experiment number"
    )

    plt.ylabel(
        "Best validation accuracy so far"
    )

    plt.title(
        "Search Efficiency by Experiment Budget"
    )

    plt.xticks(
        range(
            1,
            max(
                row["trial"]
                for row in curve_rows
            )
            + 1,
        )
    )

    if methods_present:
        plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "best_validation_vs_trial.png"
    )

    plt.savefig(
        output,
        dpi=300,
    )

    plt.close()

    print(
        f"Saved {output}"
    )


def plot_runtime(curve_rows):
    plt.figure(
        figsize=(8, 5)
    )

    methods_present = []

    for method in METHOD_ORDER:
        points = [
            row
            for row in curve_rows
            if row["search_method"] == method
        ]

        if not points:
            continue

        methods_present.append(method)

        x = [
            row[
                "cumulative_runtime_hours"
            ]
            for row in points
        ]

        y = [
            row["best_so_far_accuracy"]
            for row in points
        ]

        plt.plot(
            x,
            y,
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Cumulative proxy training time (hours)"
    )

    plt.ylabel(
        "Best validation accuracy so far"
    )

    plt.title(
        "Search Efficiency by Compute Time"
    )

    if methods_present:
        plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "best_validation_vs_runtime.png"
    )

    plt.savefig(
        output,
        dpi=300,
    )

    plt.close()

    print(
        f"Saved {output}"
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing results file: "
            f"{RESULTS_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_csv(
        RESULTS_PATH
    )

    curve_rows, summaries = (
        build_curves(rows)
    )

    if not curve_rows:
        print(
            "No completed proxy-search "
            "experiments found."
        )
        return

    write_csv(
        OUTPUT_DIR
        / "search_curves.csv",
        curve_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "search_summary.csv",
        summaries,
    )

    print_summary(
        summaries
    )

    plot_trials(
        curve_rows
    )

    plot_runtime(
        curve_rows
    )

    print(
        "\nAnalysis complete."
    )


if __name__ == "__main__":
    main()
