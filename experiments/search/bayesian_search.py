import argparse
import csv
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


def read_rows() -> list[dict[str, str]]:
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


def build_configuration_lookup(
) -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    dict[tuple[Any, ...], str],
]:
    configurations = generate_all_configurations(
        runnable_only=True
    )

    config_ids = []
    config_by_id = {}
    id_by_key = {}

    for index, config in enumerate(
        configurations,
        start=1,
    ):
        config_id = f"config_{index:03d}"

        config_ids.append(config_id)
        config_by_id[config_id] = config
        id_by_key[configuration_key(config)] = config_id

    return config_ids, config_by_id, id_by_key


def next_experiment_number(
    rows: list[dict[str, str]],
) -> int:
    numbers = []

    for row in rows:
        experiment_id = row.get("experiment_id", "")

        if not experiment_id.startswith("bayes_"):
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


def get_completed_observations(
    rows: list[dict[str, str]],
    id_by_key: dict[tuple[Any, ...], str],
    metric: str,
) -> list[tuple[str, float]]:
    observations = []

    for row in rows:
        if row.get("status") != "completed":
            continue

        metric_value = row.get(metric, "").strip()

        if not metric_value:
            continue

        config_id = id_by_key.get(row_key(row))

        if config_id is None:
            continue

        try:
            score = float(metric_value)
        except ValueError:
            continue

        observations.append((config_id, score))

    return observations


def get_tested_config_ids(
    rows: list[dict[str, str]],
    id_by_key: dict[tuple[Any, ...], str],
) -> set[str]:
    tested_ids = set()

    for row in rows:
        config_id = id_by_key.get(row_key(row))

        if config_id is not None:
            tested_ids.add(config_id)

    return tested_ids


def suggest_configuration(
    rows: list[dict[str, str]],
    seed: int,
    metric: str,
) -> tuple[str, dict[str, Any], int]:
    try:
        import optuna
        from optuna.distributions import CategoricalDistribution
        from optuna.trial import create_trial
    except ImportError as error:
        raise RuntimeError(
            "Optuna is not installed in this Python environment. "
            "Run this script from the Poetry environment on the "
            "computer used for experiments."
        ) from error

    (
        config_ids,
        config_by_id,
        id_by_key,
    ) = build_configuration_lookup()

    tested_ids = get_tested_config_ids(
        rows=rows,
        id_by_key=id_by_key,
    )

    available_ids = [
        config_id
        for config_id in config_ids
        if config_id not in tested_ids
    ]

    if not available_ids:
        raise RuntimeError(
            "There are no untested runnable configurations left."
        )

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    distribution = CategoricalDistribution(
        choices=config_ids
    )

    observations = get_completed_observations(
        rows=rows,
        id_by_key=id_by_key,
        metric=metric,
    )

    for config_id, score in observations:
        completed_trial = create_trial(
            params={
                "configuration_id": config_id,
            },
            distributions={
                "configuration_id": distribution,
            },
            value=score,
        )

        study.add_trial(completed_trial)

    max_attempts = len(config_ids) * 5

    for _ in range(max_attempts):
        trial = study.ask()

        config_id = trial.suggest_categorical(
            "configuration_id",
            config_ids,
        )

        if config_id in tested_ids:
            study.tell(
                trial,
                state=optuna.trial.TrialState.PRUNED,
            )
            continue

        return (
            config_id,
            config_by_id[config_id],
            len(observations),
        )

    raise RuntimeError(
        "Optuna repeatedly suggested configurations that "
        "have already been tested."
    )


def make_result_row(
    config: dict[str, Any],
    experiment_number: int,
) -> dict[str, str]:
    validate_configuration(config)

    return {
        "experiment_id": f"bayes_{experiment_number:03d}",
        "search_method": "bayesian",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use Optuna to add the next Bayesian-search "
            "experiment to the shared results file."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by the Optuna TPE sampler.",
    )

    parser.add_argument(
        "--metric",
        choices=[
            "val_accuracy",
            "val_f1",
        ],
        default="val_accuracy",
        help=(
            "Validation metric used to guide the search. "
            "The default is val_accuracy."
        ),
    )

    args = parser.parse_args()

    rows = read_rows()

    (
        config_id,
        config,
        observation_count,
    ) = suggest_configuration(
        rows=rows,
        seed=args.seed,
        metric=args.metric,
    )

    experiment_number = next_experiment_number(rows)

    result_row = make_result_row(
        config=config,
        experiment_number=experiment_number,
    )

    rows.append(result_row)
    write_rows(rows)

    print(
        f"Completed observations available to Optuna: "
        f"{observation_count}"
    )

    print(f"Selected configuration: {config_id}")
    print(
        f"Setting: {config['model_setting']} / "
        f"{config['profile_setting']}"
    )
    print(f"Configuration: {config}")
    print(f"Experiment: {result_row['experiment']}")
    print(f"Added as: {result_row['experiment_id']}")
    print(f"\nResults file:\n{RESULTS_PATH}")


if __name__ == "__main__":
    main()