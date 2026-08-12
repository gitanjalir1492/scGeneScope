import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import os

from results_io import atomic_write_csv


SEARCH_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SEARCH_DIRECTORY.parent.parent

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "scgenescope"
    / "scripts"
    / "train.py"
)

RESULTS_PATH = Path(
    os.environ.get(
        "SCGENESCOPE_RESULTS_PATH",
        SEARCH_DIRECTORY / "results" / "master_results.csv",
    )
).resolve()
LOG_DIRECTORY = SEARCH_DIRECTORY / "logs"
RUN_DIRECTORY = SEARCH_DIRECTORY / "runs"

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


def read_results() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file does not exist: {RESULTS_PATH}"
        )

    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        rows = list(csv.DictReader(csvfile))

    for row in rows:
        for fieldname in FIELDNAMES:
            row.setdefault(fieldname, "")

    return rows


def write_results(
    rows: list[dict[str, str]],
) -> None:
    atomic_write_csv(
        path=RESULTS_PATH,
        fieldnames=FIELDNAMES,
        rows=rows,
    )


def build_command(
    experiment: str,
    python_executable: str,
    run_directory: Path,
    proxy: bool = False,
) -> list[str]:
    command = [
        python_executable,
        str(TRAIN_SCRIPT),
        f"experiment={experiment}",
        f"hydra.run.dir={run_directory}",
        "test=false",
    ]

    if proxy:
        command.extend(
            [
                "trainer.max_epochs=10",
                "trainer.min_epochs=1",
                "callbacks.early_stopping.patience=2",
            ]
        )

    return command


def select_planned_experiment(
    rows: list[dict[str, str]],
    experiment_id: str | None = None,
) -> dict[str, str] | None:
    if experiment_id is not None:
        matching_rows = [
            row
            for row in rows
            if row.get("experiment_id") == experiment_id
        ]

        if not matching_rows:
            raise ValueError(
                "No experiment was found with ID "
                f"'{experiment_id}'."
            )

        row = matching_rows[0]

        if row.get("status") != "planned":
            raise ValueError(
                f"Experiment '{experiment_id}' has status "
                f"'{row.get('status', '')}', not 'planned'."
            )

        return row

    for row in rows:
        if row.get("status") == "planned":
            return row

    return None


def find_metrics_file(
    run_directory: Path,
) -> Path:
    metrics_files = list(
        run_directory.rglob("metrics.csv")
    )

    if not metrics_files:
        raise FileNotFoundError(
            "Training finished, but no metrics.csv file "
            f"was found under {run_directory}."
        )

    return max(
        metrics_files,
        key=lambda path: path.stat().st_mtime,
    )


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def read_validation_metrics(
    metrics_path: Path,
) -> tuple[float, float | None]:
    with metrics_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        metric_rows = list(
            csv.DictReader(csvfile)
        )

    best_accuracy = None
    best_f1 = None
    all_f1_values = []

    for metric_row in metric_rows:
        accuracy = parse_float(
            metric_row.get("val/acc")
        )

        f1 = parse_float(
            metric_row.get("val/f1")
        )

        if f1 is not None:
            all_f1_values.append(f1)

        if accuracy is None:
            continue

        if (
            best_accuracy is None
            or accuracy > best_accuracy
        ):
            best_accuracy = accuracy
            best_f1 = f1

    if best_accuracy is None:
        raise ValueError(
            "metrics.csv does not contain a valid "
            "'val/acc' value."
        )

    if best_f1 is None and all_f1_values:
        best_f1 = max(all_f1_values)

    return best_accuracy, best_f1


def make_run_directory(
    experiment_id: str,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        RUN_DIRECTORY
        / f"{experiment_id}_{timestamp}"
    ).resolve()


def run_experiment(
    row: dict[str, str],
    rows: list[dict[str, str]],
    python_executable: str,
    proxy: bool = False,
) -> None:
    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_id = row["experiment_id"]

    log_path = (
        LOG_DIRECTORY
        / f"{experiment_id}.log"
    ).resolve()

    run_directory = make_run_directory(
        experiment_id
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    command = build_command(
        experiment=row["experiment"],
        python_executable=python_executable,
        run_directory=run_directory,
        proxy=proxy,
    )

    print("\nSelected experiment:")
    print(f"ID: {experiment_id}")
    print(f"Method: {row['search_method']}")
    print(
        "Setting: "
        f"{row['model_setting']} / "
        f"{row['profile_setting']}"
    )
    print(f"Experiment: {row['experiment']}")

    if proxy:
        print("Mode: PROXY")
        print(
            "Proxy overrides: "
            "max_epochs=10, min_epochs=1, "
            "early_stopping.patience=2"
        )

    print("\nCommand that will be executed:\n")
    print(" ".join(command))

    print("\nRun directory:")
    print(run_directory)

    print("\nLog file:")
    print(log_path)

    row["status"] = "running"
    row["started_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    row["finished_at"] = ""
    row["return_code"] = ""
    row["run_directory"] = str(run_directory)
    row["log_file"] = str(log_path)
    row["error_message"] = ""

    write_results(rows)

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as logfile:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            text=True,
        )

    row["return_code"] = str(
        result.returncode
    )
    row["finished_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    if result.returncode != 0:
        row["status"] = "failed"
        row["error_message"] = (
            f"Training returned exit code "
            f"{result.returncode}."
        )
        write_results(rows)

        raise RuntimeError(
            f"Experiment {experiment_id} failed "
            f"with exit code {result.returncode}."
        )

    metrics_path = find_metrics_file(
        run_directory
    )

    val_accuracy, val_f1 = (
        read_validation_metrics(
            metrics_path
        )
    )

    row["status"] = "completed"
    row["val_accuracy"] = str(
        val_accuracy
    )
    row["val_f1"] = (
        ""
        if val_f1 is None
        else str(val_f1)
    )
    row["error_message"] = ""

    write_results(rows)

    print(
        f"\nExperiment {experiment_id} "
        "completed successfully."
    )
    print(
        f"Validation accuracy: "
        f"{val_accuracy}"
    )
    print(
        f"Validation F1: "
        f"{val_f1}"
    )
    print(
        f"Metrics file: "
        f"{metrics_path}"
    )


def preview_experiment(
    row: dict[str, str],
    python_executable: str,
    proxy: bool = False,
) -> None:
    preview_directory = (
        RUN_DIRECTORY
        / f"{row['experiment_id']}_TIMESTAMP"
    ).resolve()

    command = build_command(
        experiment=row["experiment"],
        python_executable=python_executable,
        run_directory=preview_directory,
        proxy=proxy,
    )

    print("\nSelected experiment:")
    print(f"ID: {row['experiment_id']}")
    print(f"Method: {row['search_method']}")
    print(
        "Setting: "
        f"{row['model_setting']} / "
        f"{row['profile_setting']}"
    )
    print(f"Experiment: {row['experiment']}")

    if proxy:
        print("Mode: PROXY")

    print("\nCommand that will be executed:\n")
    print(" ".join(command))

    print(
        "\nNote: test=false is used during "
        "search. The held-out test set is "
        "evaluated only after selecting the "
        "best model."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or run a planned "
            "scGeneScope experiment."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the selected planned experiment. "
            "Without this flag, only show the command."
        ),
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python executable used to start "
            "training."
        ),
    )

    parser.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "ID of a specific planned experiment "
            "to preview or run. If omitted, the "
            "first planned row is selected."
        ),
    )

    parser.add_argument(
        "--proxy",
        action="store_true",
        help=(
            "Run a shortened proxy evaluation "
            "using 10 max epochs and early "
            "stopping patience 2."
        ),
    )

    args = parser.parse_args()

    rows = read_results()

    row = select_planned_experiment(
        rows=rows,
        experiment_id=args.experiment_id,
    )

    if row is None:
        print(
            "No planned experiments were found."
        )
        return

    if args.execute:
        run_experiment(
            row=row,
            rows=rows,
            python_executable=args.python,
            proxy=args.proxy,
        )
    else:
        preview_experiment(
            row=row,
            python_executable=args.python,
            proxy=args.proxy,
        )


if __name__ == "__main__":
    main()
