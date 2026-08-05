import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SEARCH_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SEARCH_DIRECTORY.parent.parent

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "scgenescope"
    / "scripts"
    / "train.py"
)

RESULTS_PATH = SEARCH_DIRECTORY / "master_results.csv"
LOG_DIRECTORY = SEARCH_DIRECTORY / "logs"

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
    "experiment_summary",
    "started_at",
    "finished_at",
    "return_code",
    "run_directory",
    "log_file",
    "error_message",
]


def read_results() -> list[dict[str, str]]:
    """Read the shared experiment results file."""

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


def write_results(rows: list[dict[str, str]]) -> None:
    """Save the current experiment records."""

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_command(
    experiment: str,
    python_executable: str,
) -> list[str]:
    return [
        python_executable,
        str(TRAIN_SCRIPT),
        f"experiment={experiment}",
    ]


def select_next_planned_experiment(
    rows: list[dict[str, str]],
) -> dict[str, str] | None:
    for row in rows:
        if row.get("status") == "planned":
            return row

    return None


def run_experiment(
    row: dict[str, str],
    rows: list[dict[str, str]],
    python_executable: str,
) -> None:
    """Run one planned experiment and record whether it succeeded."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    experiment_id = row["experiment_id"]
    log_path = LOG_DIRECTORY / f"{experiment_id}.log"

    command = build_command(
        experiment=row["experiment"],
        python_executable=python_executable,
    )

    print("\nExecuting command:\n")
    print(subprocess.list2cmdline(command))
    print(f"\nLog file:\n{log_path}\n")

    row["status"] = "running"
    row["started_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    row["finished_at"] = ""
    row["return_code"] = ""
    row["run_directory"] = ""
    row["log_file"] = str(log_path)
    row["error_message"] = ""

    write_results(rows)

    try:
        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        row["return_code"] = str(result.returncode)
        row["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )

        if result.returncode == 0:
            row["status"] = "completed"
            print(
                f"Experiment {experiment_id} completed successfully."
            )
        else:
            row["status"] = "failed"
            row["error_message"] = (
                f"Training returned exit code "
                f"{result.returncode}."
            )

            print(
                f"Experiment {experiment_id} failed with "
                f"exit code {result.returncode}."
            )

    except Exception as error:
        row["status"] = "failed"
        row["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        row["error_message"] = str(error)

        print(
            f"Experiment {experiment_id} could not be run:"
        )
        print(error)

    finally:
        write_results(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or run the next planned scGeneScope experiment."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the next planned experiment. "
            "Without this flag, only show the command."
        ),
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python executable used to start training. "
            "Defaults to the current Python interpreter."
        ),
    )

    args = parser.parse_args()

    rows = read_results()

    planned_count = sum(
        row.get("status") == "planned"
        for row in rows
    )

    print(f"Project root:\n{PROJECT_ROOT}\n")
    print(f"Training script:\n{TRAIN_SCRIPT}\n")
    print(f"Python:\n{args.python}\n")
    print(f"Found {planned_count} planned experiments.\n")

    row = select_next_planned_experiment(rows)

    if row is None:
        print("There are no planned experiments to run.")
        return

    command = build_command(
        experiment=row["experiment"],
        python_executable=args.python,
    )

    print("Next experiment:")
    print(f"ID: {row['experiment_id']}")
    print(f"Method: {row['search_method']}")
    print(
        f"Setting: {row['model_setting']} / "
        f"{row['profile_setting']}"
    )
    print(f"Experiment: {row['experiment']}")
    print("\nCommand that will be executed:\n")
    print(subprocess.list2cmdline(command))

    if not args.execute:
        print("\nDry run complete. No experiment was started.")
        return

    run_experiment(
        row=row,
        rows=rows,
        python_executable=args.python,
    )


if __name__ == "__main__":
    main()