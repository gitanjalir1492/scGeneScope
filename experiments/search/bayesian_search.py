import argparse
import csv
from pathlib import Path
from typing import Any

from search_space import (
    AGGREGATIONS,
    IMAGE_ENCODERS,
    MODEL_SETTINGS,
    PROFILE_SETTINGS,
    RNA_ENCODERS,
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


def write_rows(
    rows: list[dict[str, str]],
) -> None:
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def get_bayesian_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("search_method") == "bayesian"
    ]


def row_to_configuration(
    row: dict[str, str],
) -> dict[str, Any]:
    return {
        "model_setting": (
            row.get("model_setting") or None
        ),
        "profile_setting": (
            row.get("profile_setting") or None
        ),
        "rna_encoder": (
            row.get("rna_encoder") or None
        ),
        "image_encoder": (
            row.get("image_encoder") or None
        ),
        "aggregation": (
            row.get("aggregation") or None
        ),
        "fusion": (
            row.get("fusion") or None
        ),
    }


def build_runnable_lookup(
) -> tuple[
    dict[tuple[Any, ...], dict[str, Any]],
    dict[tuple[Any, ...], str],
]:
    configurations = generate_all_configurations(
        runnable_only=True
    )

    config_by_key = {}
    id_by_key = {}

    for index, config in enumerate(
        configurations,
        start=1,
    ):
        key = configuration_key(config)

        config_by_key[key] = config
        id_by_key[key] = f"config_{index:03d}"

    return config_by_key, id_by_key


def next_experiment_number(
    rows: list[dict[str, str]],
) -> int:
    numbers = []

    for row in get_bayesian_rows(rows):
        experiment_id = row.get(
            "experiment_id",
            "",
        )

        if not experiment_id.startswith(
            "bayes_"
        ):
            continue

        try:
            numbers.append(
                int(
                    experiment_id.split("_")[-1]
                )
            )
        except ValueError:
            continue

    return max(numbers, default=0) + 1


def suggest_parameters(
    trial,
) -> dict[str, Any]:
    """
    Ask Optuna for one runnable configuration.

    Conditional branches expose only parameters that genuinely vary
    within the currently runnable search space. Fixed implementation
    choices are filled in directly rather than sampled.
    """

    model_setting = trial.suggest_categorical(
        "model_setting",
        MODEL_SETTINGS,
    )

    profile_setting = trial.suggest_categorical(
        "profile_setting",
        PROFILE_SETTINGS,
    )

    config = {
        "model_setting": model_setting,
        "profile_setting": profile_setting,
        "rna_encoder": None,
        "image_encoder": None,
        "aggregation": None,
        "fusion": None,
    }

    if model_setting == "rna_only":
        config["rna_encoder"] = (
            trial.suggest_categorical(
                "rna_encoder",
                RNA_ENCODERS,
            )
        )

        if profile_setting == "multiprofile":
            config["aggregation"] = (
                trial.suggest_categorical(
                    "aggregation",
                    AGGREGATIONS,
                )
            )

    elif model_setting == "image_only":
        if profile_setting == "singleprofile":
            config["image_encoder"] = (
                trial.suggest_categorical(
                    "image_encoder",
                    IMAGE_ENCODERS,
                )
            )

        else:
            # ResNet50 multiprofile configs are not currently
            # implemented, so ViT-L is the only runnable encoder.
            config["image_encoder"] = (
                "imagenet_vit_l"
            )

            config["aggregation"] = (
                trial.suggest_categorical(
                    "aggregation",
                    AGGREGATIONS,
                )
            )

    elif model_setting == "multimodal":
        config["rna_encoder"] = (
            trial.suggest_categorical(
                "rna_encoder",
                RNA_ENCODERS,
            )
        )

        # These are currently the only implemented multimodal
        # representation and fusion choices.
        config["image_encoder"] = (
            "imagenet_vit_l"
        )
        config["fusion"] = "concat"

        if profile_setting == "multiprofile":
            # Multimodal Transformer aggregation is not yet
            # implemented, so avgpool is currently fixed.
            config["aggregation"] = "avgpool"

    validate_configuration(config)

    runnable_keys = {
        configuration_key(candidate)
        for candidate
        in generate_all_configurations(
            runnable_only=True
        )
    }

    if configuration_key(config) not in runnable_keys:
        raise RuntimeError(
            "Bayesian sampler produced a configuration "
            "outside the runnable search space. "
            f"Configuration: {config}"
        )

    return config


def trial_data_for_configuration(
    config: dict[str, Any],
):
    """
    Reconstruct the Optuna parameterization used for a completed
    runnable configuration.

    Only parameters that vary within that conditional branch are
    included. Fixed implementation choices are deliberately omitted.
    """

    from optuna.distributions import (
        CategoricalDistribution,
    )

    params = {
        "model_setting": (
            config["model_setting"]
        ),
        "profile_setting": (
            config["profile_setting"]
        ),
    }

    distributions = {
        "model_setting": (
            CategoricalDistribution(
                choices=MODEL_SETTINGS
            )
        ),
        "profile_setting": (
            CategoricalDistribution(
                choices=PROFILE_SETTINGS
            )
        ),
    }

    model_setting = config[
        "model_setting"
    ]
    profile_setting = config[
        "profile_setting"
    ]

    if model_setting == "rna_only":
        params["rna_encoder"] = (
            config["rna_encoder"]
        )

        distributions["rna_encoder"] = (
            CategoricalDistribution(
                choices=RNA_ENCODERS
            )
        )

        if profile_setting == "multiprofile":
            params["aggregation"] = (
                config["aggregation"]
            )

            distributions["aggregation"] = (
                CategoricalDistribution(
                    choices=AGGREGATIONS
                )
            )

    elif model_setting == "image_only":
        if profile_setting == "singleprofile":
            params["image_encoder"] = (
                config["image_encoder"]
            )

            distributions["image_encoder"] = (
                CategoricalDistribution(
                    choices=IMAGE_ENCODERS
                )
            )

        else:
            params["aggregation"] = (
                config["aggregation"]
            )

            distributions["aggregation"] = (
                CategoricalDistribution(
                    choices=AGGREGATIONS
                )
            )

    elif model_setting == "multimodal":
        params["rna_encoder"] = (
            config["rna_encoder"]
        )

        distributions["rna_encoder"] = (
            CategoricalDistribution(
                choices=RNA_ENCODERS
            )
        )

    return params, distributions


def add_completed_trials(
    study,
    rows: list[dict[str, str]],
    metric: str,
) -> int:
    from optuna.trial import create_trial

    completed_count = 0

    runnable_keys = {
        configuration_key(config)
        for config
        in generate_all_configurations(
            runnable_only=True
        )
    }

    for row in get_bayesian_rows(rows):
        if row.get("status") != "completed":
            continue

        metric_value = row.get(
            metric,
            "",
        ).strip()

        if not metric_value:
            continue

        try:
            score = float(metric_value)
        except ValueError:
            continue

        config = row_to_configuration(
            row
        )

        try:
            validate_configuration(
                config
            )
        except ValueError:
            continue

        if (
            configuration_key(config)
            not in runnable_keys
        ):
            continue

        params, distributions = (
            trial_data_for_configuration(
                config
            )
        )

        completed_trial = create_trial(
            params=params,
            distributions=distributions,
            value=score,
        )

        study.add_trial(
            completed_trial
        )

        completed_count += 1

    return completed_count


def get_tested_keys(
    rows: list[dict[str, str]],
) -> set[tuple[Any, ...]]:
    return {
        row_key(row)
        for row in get_bayesian_rows(rows)
    }


def suggest_configuration(
    rows: list[dict[str, str]],
    seed: int,
    metric: str,
    startup_trials: int,
) -> tuple[
    str,
    dict[str, Any],
    int,
]:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna is not installed. Run this "
            "script from the project Poetry environment."
        ) from error

    config_by_key, id_by_key = (
        build_runnable_lookup()
    )

    tested_keys = get_tested_keys(
        rows
    )

    available_keys = (
        set(config_by_key)
        - tested_keys
    )

    if not available_keys:
        raise RuntimeError(
            "There are no untested runnable Bayesian "
            "configurations left."
        )

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=startup_trials,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    completed_count = add_completed_trials(
        study=study,
        rows=rows,
        metric=metric,
    )

    maximum_attempts = max(
        len(config_by_key) * 5,
        50,
    )

    for _ in range(
        maximum_attempts
    ):
        trial = study.ask()

        config = suggest_parameters(
            trial
        )

        key = configuration_key(
            config
        )

        if key in tested_keys:
            study.tell(
                trial,
                state=(
                    optuna.trial.TrialState.PRUNED
                ),
            )

            continue

        return (
            id_by_key[key],
            config_by_key[key],
            completed_count,
        )

    raise RuntimeError(
        "Optuna did not find an untested runnable "
        "configuration after repeated attempts."
    )


def make_result_row(
    config: dict[str, Any],
    experiment_number: int,
    completed_count: int,
) -> dict[str, str]:
    validate_configuration(
        config
    )

    return {
        "experiment_id": (
            f"bayes_{experiment_number:03d}"
        ),
        "search_method": "bayesian",
        "model_setting": (
            config["model_setting"]
        ),
        "profile_setting": (
            config["profile_setting"]
        ),
        "rna_encoder": (
            config["rna_encoder"] or ""
        ),
        "image_encoder": (
            config["image_encoder"] or ""
        ),
        "aggregation": (
            config["aggregation"] or ""
        ),
        "fusion": (
            config["fusion"] or ""
        ),
        "experiment": (
            build_experiment_name(
                config
            )
        ),
        "status": "planned",
        "val_accuracy": "",
        "val_f1": "",
        "test_accuracy": "",
        "test_f1": "",
        "selection_reason": (
            "Optuna TPE suggestion based on "
            f"{completed_count} completed Bayesian "
            "trial(s), restricted to the currently "
            "runnable search space."
        ),
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
            "Add the next runnable component-level "
            "Optuna suggestion to the shared "
            "results file."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed used by the Optuna TPE sampler."
        ),
    )

    parser.add_argument(
        "--metric",
        choices=[
            "val_accuracy",
            "val_f1",
        ],
        default="val_accuracy",
        help=(
            "Validation metric used by Optuna."
        ),
    )

    parser.add_argument(
        "--startup-trials",
        type=int,
        default=3,
        help=(
            "Number of initial observations before "
            "TPE begins using its fitted model."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate and print the next suggestion "
            "without modifying master_results.csv."
        ),
    )

    args = parser.parse_args()

    if args.startup_trials < 1:
        raise ValueError(
            "--startup-trials must be at least 1."
        )

    rows = read_rows()

    (
        config_id,
        config,
        completed_count,
    ) = suggest_configuration(
        rows=rows,
        seed=args.seed,
        metric=args.metric,
        startup_trials=args.startup_trials,
    )

    experiment_number = (
        next_experiment_number(
            rows
        )
    )

    result_row = make_result_row(
        config=config,
        experiment_number=experiment_number,
        completed_count=completed_count,
    )

    print(
        "Completed Bayesian observations available "
        f"to Optuna: {completed_count}"
    )

    print(
        f"Selected configuration: {config_id}"
    )

    print(
        f"Setting: "
        f"{config['model_setting']} / "
        f"{config['profile_setting']}"
    )

    print(
        f"Configuration: {config}"
    )

    print(
        f"Experiment: "
        f"{result_row['experiment']}"
    )

    if args.dry_run:
        print(
            "\nDry run complete. "
            "No experiment was added."
        )

        return

    rows.append(
        result_row
    )

    write_rows(
        rows
    )

    print(
        f"Added as: "
        f"{result_row['experiment_id']}"
    )

    print(
        f"\nResults file:\n"
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
