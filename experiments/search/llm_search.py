import argparse
import csv
import json
from pathlib import Path

from search_space import (
    build_experiment_name,
    generate_all_configurations,
    validate_configuration,
)


SEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SEARCH_DIR / "master_results.csv"

FIELDS = [
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


def load_results():
    if not RESULTS_PATH.exists():
        return []

    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        for field in FIELDS:
            row.setdefault(field, "")

    return rows


def save_results(rows):
    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def config_key(config):
    return (
        config["model_setting"],
        config["profile_setting"],
        config["rna_encoder"],
        config["image_encoder"],
        config["aggregation"],
        config["fusion"],
    )


def row_key(row):
    return (
        row.get("model_setting") or None,
        row.get("profile_setting") or None,
        row.get("rna_encoder") or None,
        row.get("image_encoder") or None,
        row.get("aggregation") or None,
        row.get("fusion") or None,
    )


def method_name(memory_mode):
    if memory_mode == "metrics":
        return "llm_metrics"

    return "llm_summary"


def build_lookup():
    configs = generate_all_configurations(
        runnable_only=True
    )

    config_by_id = {}
    id_by_key = {}

    for index, config in enumerate(
        configs,
        start=1,
    ):
        config_id = f"config_{index:03d}"
        config_by_id[config_id] = config
        id_by_key[config_key(config)] = config_id

    return config_by_id, id_by_key


def format_config(config_id, config):
    return (
        f"{config_id}: "
        f"model_setting={config['model_setting']}, "
        f"profile_setting={config['profile_setting']}, "
        f"rna_encoder={config['rna_encoder']}, "
        f"image_encoder={config['image_encoder']}, "
        f"aggregation={config['aggregation']}, "
        f"fusion={config['fusion']}"
    )


def format_history(row, config_id, memory_mode):
    text = (
        f"Configuration: {config_id}\n"
        f"Model setting: {row.get('model_setting')}\n"
        f"Profile setting: {row.get('profile_setting')}\n"
        f"RNA encoder: {row.get('rna_encoder') or None}\n"
        f"Image encoder: {row.get('image_encoder') or None}\n"
        f"Aggregation: {row.get('aggregation') or None}\n"
        f"Fusion: {row.get('fusion') or None}\n"
        f"Validation accuracy: "
        f"{row.get('val_accuracy') or 'not available'}\n"
        f"Validation F1: "
        f"{row.get('val_f1') or 'not available'}"
    )

    if memory_mode == "summary":
        text += (
            "\nExperiment summary: "
            f"{row.get('experiment_summary') or 'not available'}"
        )

    return text


def build_prompt(rows, memory_mode):
    search_method = method_name(memory_mode)
    config_by_id, id_by_key = build_lookup()

    method_rows = [
        row
        for row in rows
        if row.get("search_method") == search_method
    ]

    used_ids = {
        id_by_key[row_key(row)]
        for row in method_rows
        if row_key(row) in id_by_key
    }

    available_ids = [
        config_id
        for config_id in config_by_id
        if config_id not in used_ids
    ]

    if not available_ids:
        raise RuntimeError(
            "No untested configurations remain."
        )

    completed_rows = [
        row
        for row in method_rows
        if row.get("status") == "completed"
    ]

    history = []

    for row in completed_rows:
        config_id = id_by_key.get(row_key(row))

        if config_id is not None:
            history.append(
                format_history(
                    row,
                    config_id,
                    memory_mode,
                )
            )

    if history:
        history_text = "\n\n".join(history)
    else:
        history_text = (
            "No experiments have been completed yet."
        )

    available_text = "\n".join(
        format_config(
            config_id,
            config_by_id[config_id],
        )
        for config_id in available_ids
    )

    prompt = f"""
You are choosing the next experiment for a model search study in
multimodal cellular profiling.

The aim is to find a strong model configuration within a limited
experiment budget. Use the previous validation results to decide which
untested configuration is most informative or most likely to improve
performance.

Important:
- Optimise validation accuracy.
- Use validation F1 as supporting evidence.
- Do not choose a configuration that has already been tested.
- Do not assume that multimodal or more complex models are always better.
- Consider what has already been learned about the representations,
  profile settings, aggregation methods and fusion methods.
- When there is little evidence, choose an experiment that will reduce
  uncertainty and help guide later choices.
- Only select from the available configuration IDs listed below.

Previous completed experiments:
{history_text}

Available untested configurations:
{available_text}

Choose exactly one configuration.

Return JSON only, using this format:
{{
  "configuration_id": "config_001",
  "reason": "Explain briefly what evidence or uncertainty motivated this choice."
}}
""".strip()

    return prompt, config_by_id, available_ids


def parse_response(text):
    try:
        response = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Response is not valid JSON."
        ) from error

    if not isinstance(response, dict):
        raise ValueError(
            "Response must be a JSON object."
        )

    config_id = response.get("configuration_id")
    reason = response.get("reason", "")

    if not isinstance(config_id, str):
        raise ValueError(
            "Missing configuration_id."
        )

    if not isinstance(reason, str):
        raise ValueError(
            "Reason must be a string."
        )

    return config_id, reason


def next_number(rows, search_method):
    if search_method == "llm_metrics":
        prefix = "llm_metrics_"
    else:
        prefix = "llm_summary_"

    numbers = []

    for row in rows:
        experiment_id = row.get(
            "experiment_id",
            "",
        )

        if not experiment_id.startswith(prefix):
            continue

        try:
            numbers.append(
                int(experiment_id.split("_")[-1])
            )
        except ValueError:
            pass

    return max(numbers, default=0) + 1


def make_row(
    config,
    search_method,
    reason,
    number,
):
    if search_method == "llm_metrics":
        prefix = "llm_metrics"
    else:
        prefix = "llm_summary"

    validate_configuration(config)

    return {
        "experiment_id": (
            f"{prefix}_{number:03d}"
        ),
        "search_method": search_method,
        "model_setting": config["model_setting"],
        "profile_setting": config["profile_setting"],
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
        "experiment": build_experiment_name(
            config
        ),
        "status": "planned",
        "val_accuracy": "",
        "val_f1": "",
        "test_accuracy": "",
        "test_f1": "",
        "selection_reason": reason,
        "experiment_summary": "",
        "started_at": "",
        "finished_at": "",
        "return_code": "",
        "run_directory": "",
        "log_file": "",
        "error_message": "",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--memory-mode",
        choices=[
            "metrics",
            "summary",
        ],
        default="metrics",
    )

    parser.add_argument(
        "--response-file",
        type=Path,
    )

    parser.add_argument(
        "--save-prompt",
        type=Path,
    )

    args = parser.parse_args()

    rows = load_results()

    (
        prompt,
        config_by_id,
        available_ids,
    ) = build_prompt(
        rows,
        args.memory_mode,
    )

    if args.save_prompt:
        args.save_prompt.write_text(
            prompt,
            encoding="utf-8",
        )

        print(
            f"Saved prompt to "
            f"{args.save_prompt}"
        )

    if args.response_file is None:
        print(prompt)
        print(
            "\nNo experiment was added."
        )
        return

    if not args.response_file.exists():
        raise FileNotFoundError(
            f"Response file not found: "
            f"{args.response_file}"
        )

    response_text = (
        args.response_file.read_text(
            encoding="utf-8",
        )
    )

    config_id, reason = parse_response(
        response_text
    )

    if config_id not in config_by_id:
        raise ValueError(
            f"Unknown configuration: {config_id}"
        )

    if config_id not in available_ids:
        raise ValueError(
            f"{config_id} has already been selected."
        )

    config = config_by_id[config_id]
    search_method = method_name(
        args.memory_mode
    )
    number = next_number(
        rows,
        search_method,
    )

    row = make_row(
        config,
        search_method,
        reason,
        number,
    )

    rows.append(row)
    save_results(rows)

    print(f"Selected: {config_id}")
    print(f"Configuration: {config}")
    print(f"Reason: {reason}")
    print(f"Experiment: {row['experiment']}")
    print(f"Added as: {row['experiment_id']}")


if __name__ == "__main__":
    main()