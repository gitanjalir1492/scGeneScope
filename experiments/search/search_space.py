import itertools
from typing import Any


SEARCH_SPACE = {
    "rna_encoder": [
        "pca_n2000",
        "scvi_n200",
        "scgpt",
    ],
    "image_encoder": [
        "imagenet_vit_l",
        "imagenet_vit_h",
    ],
    "fusion": [
        "concat",
    ],
    "aggregation": [
        "singleprofile",
        "avgpool",
    ],
}


def generate_all_configurations() -> list[dict[str, Any]]:
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[key] for key in keys]

    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]


def build_experiment_name(config: dict[str, Any]) -> str:
    aggregation = config["aggregation"]
    rna_encoder = config["rna_encoder"]
    image_encoder = config["image_encoder"]
    fusion = config["fusion"]

    if aggregation == "singleprofile":
        return (
            "multimodal/singleprofile/"
            f"train_on_{rna_encoder}"
            f"_with_{fusion}_{image_encoder}"
        )

    if aggregation == "avgpool":
        return (
            "multimodal/multiprofile/"
            f"train_avgpool_on_{rna_encoder}"
            f"_with_{fusion}_{image_encoder}"
        )

    raise ValueError(
        f"Unsupported aggregation method: {aggregation}"
    )


def validate_configuration(config: dict[str, Any]) -> None:
    required_keys = set(SEARCH_SPACE.keys())
    provided_keys = set(config.keys())

    if provided_keys != required_keys:
        missing = required_keys - provided_keys
        extra = provided_keys - required_keys

        raise ValueError(
            f"Invalid configuration keys. "
            f"Missing: {sorted(missing)}. "
            f"Extra: {sorted(extra)}."
        )

    for key, value in config.items():
        if value not in SEARCH_SPACE[key]:
            raise ValueError(
                f"Invalid value for {key}: {value}. "
                f"Allowed values: {SEARCH_SPACE[key]}"
            )


if __name__ == "__main__":
    configurations = generate_all_configurations()

    print(
        f"Search space contains "
        f"{len(configurations)} unique configurations."
    )

    for config in configurations:
        validate_configuration(config)
        print(build_experiment_name(config))