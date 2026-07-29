import csv
import random
from pathlib import Path

from search_space import (
    build_experiment_name,
    generate_all_configurations,
    validate_configuration,
)

RANDOM_SEED = 42
NUM_EXPERIMENTS = 10
RESULTS_PATH = Path(__file__).parent / "master_results.csv"

FIELDNAMES = [
    "experiment_id",
    "search_method",
    "rna_encoder",
    "image_encoder",
    "fusion",
    "aggregation",
    "experiment",
    "status",
]


def write_experiment_plan(
    sampled_configurations: list[dict[str, str]],
) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for index, config in enumerate(
            sampled_configurations,
            start=1,
        ):
            validate_configuration(config)
            experiment = build_experiment_name(config)

            writer.writerow(
                {
                    "experiment_id": f"random_{index:03d}",
                    "search_method": "random",
                    "rna_encoder": config["rna_encoder"],
                    "image_encoder": config["image_encoder"],
                    "fusion": config["fusion"],
                    "aggregation": config["aggregation"],
                    "experiment": experiment,
                    "status": "planned",
                }
            )


def main() -> None:
    random.seed(RANDOM_SEED)

    all_configurations = generate_all_configurations()

    if NUM_EXPERIMENTS > len(all_configurations):
        raise ValueError(
            f"Requested {NUM_EXPERIMENTS} experiments, but the "
            f"search space contains only "
            f"{len(all_configurations)} unique configurations."
        )

    sampled_configurations = random.sample(
        all_configurations,
        NUM_EXPERIMENTS,
    )

    write_experiment_plan(sampled_configurations)

    print(
        f"Search space contains "
        f"{len(all_configurations)} unique configurations."
    )

    for index, config in enumerate(
        sampled_configurations,
        start=1,
    ):
        experiment = build_experiment_name(config)

        print(f"\nExperiment {index}")
        print(f"Configuration: {config}")
        print(f"python train.py experiment={experiment}")

    print(
        f"\nSaved {NUM_EXPERIMENTS} planned experiments to:"
    )
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()