import argparse
import csv
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from search_space import (
    build_experiment_name,
    generate_all_configurations,
    validate_configuration,
)


SEARCH_DIRECTORY = Path(__file__).resolve().parent
RESULTS_PATH = SEARCH_DIRECTORY / "master_results.csv"

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


def read_existing_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []

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


def write_rows(rows: list[dict[str, str]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

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


def configuration_key(
    config: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        config["model_setting"],
        config["profile_setting"],
        config["rna_encoder"],
        config["image_encoder"],
        config["aggregation"],
        config["fusion"],
    )


def row_key(
    row: dict[str, str],
) -> tuple[Any, ...]:
    return (
        row.get("model_setting") or None,
        row.get("profile_setting") or None,
        row.get("rna_encoder") or None,
        row.get("image_encoder") or None,
        row.get("aggregation") or None,
        row.get("fusion") or None,
    )


def next_experiment_number(
    rows: list[dict[str, str]],
) -> int:
    numbers = []

    for row in rows:
        experiment_id = row.get("experiment_id", "")

        if not experiment_id.startswith("random_"):
            continue

        try:
            numbers.append(
                int(experiment_id.split("_")[-1])
            )
        except ValueError:
            continue

    if not numbers:
        return 1

    return max(numbers) + 1


def make_result_row(
    config: dict[str, Any],
    experiment_number: int,
) -> dict[str, str]:
    validate_configuration(config)

    return {
        "experiment_id": f"random_{experiment_number:03d}",
        "search_method": "random",
        "model_setting": config["model_setting"],
        "profile_setting": config["profile_setting"],
        "rna_encoder": config["rna_encoder"] or "",
        "image_encoder": config["image_encoder"] or "",
        "aggregation": config["aggregation"] or "",
        "fusion": config["fusion"] or "",
        "experiment": build_experiment_name(config),
        "status": "planned",
        "val_accuracy": "",
        "val_f1": "",
        "test_accuracy": "",
        "test_f1": "",
        "selection_reason": "",
        "experiment_summary": "",
        "started_at": "",
        "finished_at": "",
        "return_code": "",
        "run_directory": "",
        "log_file": "",
        "error_message": "",
    }


def back_up_results() -> None:
    if not RESULTS_PATH.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = (
        SEARCH_DIRECTORY
        / f"master_results_backup_{timestamp}.csv"
    )

    shutil.copy2(RESULTS_PATH, backup_path)

    print(f"Backed up existing results to:\n{backup_path}\n")


def select_configurations(
    rows: list[dict[str, str]],
    number_to_select: int,
    seed: int,
) -> list[dict[str, Any]]:
    runnable_configurations = generate_all_configurations(
        runnable_only=True
    )

    existing_keys = {
        row_key(row)
        for row in rows
    }

    available_configurations = [
        config
        for config in runnable_configurations
        if configuration_key(config) not in existing_keys
    ]

    if number_to_select > len(available_configurations):
        raise ValueError(
            f"Requested {number_to_select} new experiments, "
            f"but only {len(available_configurations)} untested "
            "runnable configurations remain."
        )

    random_generator = random.Random(seed)

    return random_generator.sample(
        available_configurations,
        number_to_select,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add randomly selected experiments to the shared "
            "experiment-results file."
        )
    )

    parser.add_argument(
        "--num-experiments",
        type=int,
        default=1,
        help=(
            "Number of new random experiments to add. "
            "The default is 1."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for experiment selection.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Back up the existing results file and create "
            "a new random-search plan."
        ),
    )

    args = parser.parse_args()

    if args.num_experiments < 1:
        raise ValueError(
            "--num-experiments must be at least 1."
        )

    if args.reset:
        back_up_results()
        rows = []
    else:
        rows = read_existing_rows()

    selected_configurations = select_configurations(
        rows=rows,
        number_to_select=args.num_experiments,
        seed=args.seed,
    )

    experiment_number = next_experiment_number(rows)

    new_rows = []

    for config in selected_configurations:
        result_row = make_result_row(
            config=config,
            experiment_number=experiment_number,
        )

        rows.append(result_row)
        new_rows.append(result_row)
        experiment_number += 1

    write_rows(rows)

    runnable_count = len(
        generate_all_configurations(runnable_only=True)
    )

    print(
        f"Runnable search space: "
        f"{runnable_count} configurations"
    )

    print(
        f"Added {len(new_rows)} random experiment(s).\n"
    )

    for row in new_rows:
        print(f"{row['experiment_id']}:")
        print(
            f"  {row['model_setting']} / "
            f"{row['profile_setting']}"
        )
        print(f"  experiment={row['experiment']}\n")

    print(f"Results file:\n{RESULTS_PATH}")


if __name__ == "__main__":
    main()