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

FIELDNAMES = [
    "experiment_id",
    "search_method",
    "rna_encoder",
    "image_encoder",
    "fusion",
    "aggregation",
    "experiment",
    "status",
    "started_at",
    "finished_at",
    "return_code",
    "run_directory",
    "error_message",
]


def read_results():
    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def build_command(experiment, python_executable):
    return [
        python_executable,
        str(TRAIN_SCRIPT),
        f"experiment={experiment}",
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
    )

    args = parser.parse_args()

    rows = read_results()

    planned = [
        row
        for row in rows
        if row["status"] == "planned"
    ]

    print(f"Project root:\n{PROJECT_ROOT}\n")
    print(f"Training script:\n{TRAIN_SCRIPT}\n")
    print(f"Python:\n{args.python}\n")
    print(f"Found {len(planned)} planned experiments.\n")

    if len(planned) == 0:
        return

    row = planned[0]

    command = build_command(
        row["experiment"],
        args.python,
    )

    print("Command that will be executed:\n")
    print(subprocess.list2cmdline(command))

    if not args.execute:
        print("\nDry run complete.")
        return

    row["status"] = "running"
    row["started_at"] = datetime.now().isoformat()

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )


if __name__ == "__main__":
    main()
