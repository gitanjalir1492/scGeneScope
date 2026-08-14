from typing import Any


MODEL_SETTINGS = [
    "rna_only",
    "image_only",
    "multimodal",
]

PROFILE_SETTINGS = [
    "singleprofile",
    "multiprofile",
]

RNA_ENCODERS = [
    "pca_n2000",
    "scvi_n200",
    "scgpt",
]

IMAGE_ENCODERS = [
    "resnet50",
    "imagenet_vit_l",
]

AGGREGATIONS = [
    "avgpool",
    "transformer",
]

FUSIONS = [
    "concat",
    "weighted",
]


def generate_target_configurations() -> list[dict[str, Any]]:
    """Create all valid configurations in the final search space."""

    configurations = []

    for rna_encoder in RNA_ENCODERS:
        configurations.append(
            {
                "model_setting": "rna_only",
                "profile_setting": "singleprofile",
                "rna_encoder": rna_encoder,
                "image_encoder": None,
                "aggregation": None,
                "fusion": None,
            }
        )

        for aggregation in AGGREGATIONS:
            configurations.append(
                {
                    "model_setting": "rna_only",
                    "profile_setting": "multiprofile",
                    "rna_encoder": rna_encoder,
                    "image_encoder": None,
                    "aggregation": aggregation,
                    "fusion": None,
                }
            )

    for image_encoder in IMAGE_ENCODERS:
        configurations.append(
            {
                "model_setting": "image_only",
                "profile_setting": "singleprofile",
                "rna_encoder": None,
                "image_encoder": image_encoder,
                "aggregation": None,
                "fusion": None,
            }
        )

        for aggregation in AGGREGATIONS:
            configurations.append(
                {
                    "model_setting": "image_only",
                    "profile_setting": "multiprofile",
                    "rna_encoder": None,
                    "image_encoder": image_encoder,
                    "aggregation": aggregation,
                    "fusion": None,
                }
            )

    for rna_encoder in RNA_ENCODERS:
        for image_encoder in IMAGE_ENCODERS:
            for fusion in FUSIONS:
                configurations.append(
                    {
                        "model_setting": "multimodal",
                        "profile_setting": "singleprofile",
                        "rna_encoder": rna_encoder,
                        "image_encoder": image_encoder,
                        "aggregation": None,
                        "fusion": fusion,
                    }
                )

            for aggregation in AGGREGATIONS:
                for fusion in FUSIONS:
                    configurations.append(
                        {
                            "model_setting": "multimodal",
                            "profile_setting": "multiprofile",
                            "rna_encoder": rna_encoder,
                            "image_encoder": image_encoder,
                            "aggregation": aggregation,
                            "fusion": fusion,
                        }
                    )

    return configurations


def validate_configuration(config: dict[str, Any]) -> None:
    """Check that a configuration follows the conditional search space."""

    required_keys = {
        "model_setting",
        "profile_setting",
        "rna_encoder",
        "image_encoder",
        "aggregation",
        "fusion",
    }

    provided_keys = set(config)

    if provided_keys != required_keys:
        missing = required_keys - provided_keys
        extra = provided_keys - required_keys

        raise ValueError(
            f"Invalid configuration keys. "
            f"Missing: {sorted(missing)}. "
            f"Extra: {sorted(extra)}."
        )

    model_setting = config["model_setting"]
    profile_setting = config["profile_setting"]
    rna_encoder = config["rna_encoder"]
    image_encoder = config["image_encoder"]
    aggregation = config["aggregation"]
    fusion = config["fusion"]

    if model_setting not in MODEL_SETTINGS:
        raise ValueError(
            f"Invalid model setting: {model_setting}"
        )

    if profile_setting not in PROFILE_SETTINGS:
        raise ValueError(
            f"Invalid profile setting: {profile_setting}"
        )

    if model_setting == "rna_only":
        if rna_encoder not in RNA_ENCODERS:
            raise ValueError(
                f"Invalid RNA encoder: {rna_encoder}"
            )

        if image_encoder is not None:
            raise ValueError(
                "RNA-only experiments cannot use an image encoder."
            )

        if fusion is not None:
            raise ValueError(
                "RNA-only experiments cannot use fusion."
            )

    if model_setting == "image_only":
        if image_encoder not in IMAGE_ENCODERS:
            raise ValueError(
                f"Invalid image encoder: {image_encoder}"
            )

        if rna_encoder is not None:
            raise ValueError(
                "Image-only experiments cannot use an RNA encoder."
            )

        if fusion is not None:
            raise ValueError(
                "Image-only experiments cannot use fusion."
            )

    if model_setting == "multimodal":
        if rna_encoder not in RNA_ENCODERS:
            raise ValueError(
                f"Invalid RNA encoder: {rna_encoder}"
            )

        if image_encoder not in IMAGE_ENCODERS:
            raise ValueError(
                f"Invalid image encoder: {image_encoder}"
            )

        if fusion not in FUSIONS:
            raise ValueError(
                f"Invalid fusion method: {fusion}"
            )

    if profile_setting == "singleprofile":
        if aggregation is not None:
            raise ValueError(
                "Single-profile experiments cannot use aggregation."
            )

    if profile_setting == "multiprofile":
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation method: {aggregation}"
            )


def get_missing_implementation(
    config: dict[str, Any],
) -> list[str]:
    """List anything that still needs to be added before a run can start."""

    validate_configuration(config)

    missing = []

    model_setting = config["model_setting"]
    profile_setting = config["profile_setting"]
    image_encoder = config["image_encoder"]
    aggregation = config["aggregation"]
    fusion = config["fusion"]

    if (
        model_setting == "multimodal"
        and profile_setting == "multiprofile"
        and aggregation == "transformer"
    ):
        missing.append(
            "multimodal Transformer aggregation"
        )

    return missing


def is_implemented(config: dict[str, Any]) -> bool:
    return not get_missing_implementation(config)


def generate_all_configurations(
    runnable_only: bool = True,
) -> list[dict[str, Any]]:
    """Return either the runnable space or the complete target space."""

    configurations = generate_target_configurations()

    if runnable_only:
        return [
            config
            for config in configurations
            if is_implemented(config)
        ]

    return configurations


def build_experiment_name(config: dict[str, Any]) -> str:
    """Convert a runnable configuration into its Hydra experiment name."""

    validate_configuration(config)

    missing = get_missing_implementation(config)

    if missing:
        raise NotImplementedError(
            "This configuration is not runnable yet. Missing: "
            + ", ".join(missing)
        )

    model_setting = config["model_setting"]
    profile_setting = config["profile_setting"]
    rna_encoder = config["rna_encoder"]
    image_encoder = config["image_encoder"]
    aggregation = config["aggregation"]
    fusion = config["fusion"]

    if model_setting == "rna_only":
        if profile_setting == "singleprofile":
            return (
                "rnaseq/singleprofile/"
                f"train_on_{rna_encoder}"
            )

        if aggregation == "avgpool":
            return (
                "rnaseq/multiprofile/"
                f"train_avgpool_on_{rna_encoder}"
            )

        return (
            "rnaseq/multiprofile/"
            f"train_transformerpool_on_{rna_encoder}"
        )

    if model_setting == "image_only":
        if profile_setting == "singleprofile":
            return (
                "imaging/singleprofile/"
                f"train_on_concat_{image_encoder}"
            )

        if aggregation == "avgpool":
            return (
                "imaging/multiprofile/"
                f"train_avgpool_on_concat_{image_encoder}"
            )

        return (
            "imaging/multiprofile/"
            f"train_transformerpool_on_concat_{image_encoder}"
        )

    if model_setting == "multimodal":
        if profile_setting == "singleprofile":
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

        return (
            "multimodal/multiprofile/"
            f"train_transformerpool_on_{rna_encoder}"
            f"_with_{fusion}_{image_encoder}"
        )

    raise ValueError(
        f"Unsupported model setting: {model_setting}"
    )


if __name__ == "__main__":
    target_configurations = generate_all_configurations(
        runnable_only=False
    )

    runnable_configurations = generate_all_configurations(
        runnable_only=True
    )

    unfinished_configurations = [
        config
        for config in target_configurations
        if not is_implemented(config)
    ]

    print(
        f"Target search space: "
        f"{len(target_configurations)} configurations"
    )

    print(
        f"Runnable now: "
        f"{len(runnable_configurations)} configurations"
    )

    print(
        f"Still to implement: "
        f"{len(unfinished_configurations)} configurations"
    )

    print("\nRunnable experiments:")

    for index, config in enumerate(
        runnable_configurations,
        start=1,
    ):
        experiment = build_experiment_name(config)

        print(f"\n{index:02d}. {config}")
        print(f"    experiment={experiment}")

    missing_items = sorted(
        {
            item
            for config in unfinished_configurations
            for item in get_missing_implementation(config)
        }
    )

    print("\nRemaining model work:")

    for item in missing_items:
        print(f"- {item}")