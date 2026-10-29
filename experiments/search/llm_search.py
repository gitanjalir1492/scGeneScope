import argparse
import csv
import json
import random
from pathlib import Path
import os

from search_space import (
    build_experiment_name,
    generate_all_configurations,
    validate_configuration,
)
from results_io import atomic_write_csv


SEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = Path(
    os.environ.get(
        "SCGENESCOPE_RESULTS_PATH",
        SEARCH_DIR / "results" / "master_results.csv",
    )
).resolve()

LLM_LOG_DIR = SEARCH_DIR / "llm_logs"

# Keep this fixed for every LLM search condition so that the only
# experimental difference is the memory supplied to the model.
LLM_MODEL = "openai_gpt54_mini"

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


def load_results() -> list[dict[str, str]]:
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


def save_results(
    rows: list[dict[str, str]],
) -> None:
    atomic_write_csv(
        path=RESULTS_PATH,
        fieldnames=FIELDS,
        rows=rows,
    )


def config_key(
    config: dict,
) -> tuple:
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
) -> tuple:
    return (
        row.get("model_setting") or None,
        row.get("profile_setting") or None,
        row.get("rna_encoder") or None,
        row.get("image_encoder") or None,
        row.get("aggregation") or None,
        row.get("fusion") or None,
    )


def method_name(
    memory_mode: str,
) -> str:
    if memory_mode == "metrics":
        return "llm_metrics"

    if memory_mode == "cost":
        return "llm_cost"

    return "llm_summary"


def build_lookup(
    shuffle_seed: int | None = None,
) -> tuple[dict, dict]:
    configs = generate_all_configurations(
        runnable_only=True
    )

    if shuffle_seed is not None:
        # IDs follow the shuffled order, so neither list position nor ID reveals the model setting
        random.Random(shuffle_seed).shuffle(configs)

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


def format_config(
    config_id: str,
    config: dict,
) -> str:
    return (
        f"{config_id}: "
        f"model_setting={config['model_setting']}, "
        f"profile_setting={config['profile_setting']}, "
        f"rna_encoder={config['rna_encoder']}, "
        f"image_encoder={config['image_encoder']}, "
        f"aggregation={config['aggregation']}, "
        f"fusion={config['fusion']}"
    )


def format_history(
    row: dict[str, str],
    config_id: str,
    memory_mode: str,
    show_gpu_hours: bool = False,
) -> str:
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

    if show_gpu_hours:
        text += (
            "\nTraining cost: "
            f"{float(row['runtime_hours']):.2f} GPU hours"
        )

    if memory_mode in {"summary", "cost"}:
        text += (
            "\nExperiment summary: "
            f"{row.get('experiment_summary') or 'not available'}"
        )

    return text


def build_prompt(
    rows: list[dict[str, str]],
    memory_mode: str,
    shuffle_seed: int | None = None,
    budget: int | None = None,
    isolation_principles: bool = True,
    gpu_hours_aware: bool = False,
) -> tuple[str, dict, list[str]]:
    """Defaults give the original prompt; shuffle_seed, budget, isolation_principles=False and gpu_hours_aware are ablation variants."""
    search_method = method_name(memory_mode)
    config_by_id, id_by_key = build_lookup(shuffle_seed)

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
        config_id = id_by_key.get(
            row_key(row)
        )

        if config_id is not None:
            history.append(
                format_history(
                    row,
                    config_id,
                    memory_mode,
                    show_gpu_hours=gpu_hours_aware,
                )
            )

    if history:
        history_text = "\n\n".join(history)
        if gpu_hours_aware:
            used_hours = sum(float(row["runtime_hours"]) for row in completed_rows)
            history_text = f"GPU hours used so far: {used_hours:.2f}\n\n{history_text}"
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

    cost_instructions = ""

    if memory_mode == "cost":
        cost_instructions = """
Additional cost-aware decision principles:
- Before choosing an experiment, estimate its relative computational
  cost from the candidate architecture itself.
- Do not assume access to measured runtime for an untested
  configuration.
- Reason about cost using architectural characteristics such as
  representation dimensionality, single- versus multiprofile
  processing, aggregation complexity, multimodal processing, and
  fusion.
- Classify the selected experiment's predicted cost as low, medium,
  or high.
- Balance expected scientific value against expected computational
  expense.
- Do not automatically prefer inexpensive experiments. A costly
  experiment is appropriate when its expected information or
  performance value justifies its expense.
"""

    if gpu_hours_aware:
        cost_instructions = """
Compute-cost principles:
- Training time is also a cost. Completed experiments show their
  measured training cost in GPU hours; untested configurations have no
  measured cost, so estimate it from the architecture (multiprofile
  aggregation, multimodal processing and larger representations take
  longer to train).
- Aim to reach the best validation accuracy while keeping the total GPU
  hours low. Prefer a cheaper configuration when its expected accuracy
  is similar to a more expensive one.
- Do not avoid an expensive configuration when it is likely to be the
  best-performing one.
- In your selection reason, also state the expected relative cost of
  the chosen experiment.
"""

    if memory_mode == "cost":
        response_format = """
{{
  "configuration_id": "config_001",
  "predicted_cost": "low",
  "cost_reason": "Briefly explain which architectural features drive the expected computational cost.",
  "reason": "Briefly explain the evidence, question being tested, and why its expected scientific value justifies this use of compute."
}}
"""
    else:
        response_format = """
{{
  "configuration_id": "config_001",
  "reason": "Briefly explain the evidence, question being tested, and why this is the best next experiment."
}}
"""

    if budget is None:
        objective_text = """
The experiment budget is limited. Your objective is therefore not only
to find a high-performing configuration, but to use each experiment
efficiently to learn which design decisions matter.
""".strip()
        informative_principle = """
- Consider whether an experiment is informative even if it may not be
  the configuration with the highest expected immediate performance."""
    else:
        experiment_number = len(completed_rows) + 1
        objective_text = (
            f"The experiment budget is {budget} experiments. This is experiment "
            f"{experiment_number} of {budget}, so {budget - experiment_number} "
            "experiment(s) will remain after this one.\n"
            "Your objective is to find the configuration with the highest "
            "validation accuracy by the end of the budget.\n"
            "Only the best validation accuracy found within the budget counts."
        )
        informative_principle = ""

    isolation_text = ""
    if isolation_principles:
        isolation_text = """
- Prefer comparisons that help isolate the effect of representation,
  profile setting, aggregation, or fusion when possible.
- Avoid changing several design choices simultaneously when a simpler
  experiment could answer the same question."""

    complexity_text = ""
    if isolation_principles:
        complexity_text = """
- Do not assume that multimodal, multiprofile, or more complex models
  will necessarily outperform simpler models."""

    prompt = f"""
You are acting as an ML research scientist choosing the next experiment
in a sequential model-search study for multimodal cellular profiling.

{objective_text}

Use the completed experiment history to decide what should be tested
next.

Decision principles:
- Validation accuracy is the primary optimization objective.
- Use validation F1 as supporting evidence, especially when accuracy
  differences are small.
- Balance exploration and exploitation.
- When evidence is weak or sparse, prefer experiments that reduce
  uncertainty about an important design choice.
- When previous experiments provide strong evidence for a promising
  direction, exploit that evidence by testing a logical extension.{isolation_text}
- Do not overinterpret a single experiment or small performance
  difference.{complexity_text}{informative_principle}
- Do not repeat a configuration that has already been selected by this
  search condition.
- Select only from the available configuration IDs listed below.

{cost_instructions}

For your selection reason, briefly state:
1. what the existing evidence or uncertainty is,
2. what this experiment will test or clarify, and
3. why it is the best use of the next experiment under the limited
   budget.

Previous completed experiments:
{history_text}

Available untested configurations:
{available_text}

Choose exactly one configuration.

Return JSON only, using exactly this format:
{response_format}
""".strip()

    return prompt, config_by_id, available_ids


def query_llm(
    prompt: str,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The OpenAI SDK is not installed. "
            "Run: poetry add openai"
        ) from error

    client = OpenAI()

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
        max_output_tokens=500,
    )

    response_text = response.output_text

    if not isinstance(
        response_text,
        str,
    ) or not response_text.strip():
        raise RuntimeError(
            "The LLM returned no text response."
        )

    return response_text.strip()


def clean_json_text(
    text: str,
) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


def parse_response(
    text: str,
    memory_mode: str,
) -> tuple[str, str, str, str]:
    cleaned_text = clean_json_text(text)

    try:
        response = json.loads(cleaned_text)
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
    predicted_cost = response.get("predicted_cost", "")
    cost_reason = response.get("cost_reason", "")

    if not isinstance(config_id, str):
        raise ValueError(
            "Missing configuration_id."
        )

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            "Reason must be a non-empty string."
        )

    if memory_mode == "cost":
        if predicted_cost not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "predicted_cost must be low, medium, or high."
            )

        if (
            not isinstance(cost_reason, str)
            or not cost_reason.strip()
        ):
            raise ValueError(
                "cost_reason must be a non-empty string."
            )

    return (
        config_id,
        reason.strip(),
        predicted_cost,
        cost_reason.strip(),
    )


def next_number(
    rows: list[dict[str, str]],
    search_method: str,
) -> int:
    if search_method == "llm_metrics":
        prefix = "llm_metrics_"
    elif search_method == "llm_cost":
        prefix = "llm_cost_"
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
                int(
                    experiment_id.split("_")[-1]
                )
            )
        except ValueError:
            pass

    return max(numbers, default=0) + 1


def make_row(
    config: dict,
    search_method: str,
    reason: str,
    number: int,
) -> dict[str, str]:
    if search_method == "llm_metrics":
        prefix = "llm_metrics"
    elif search_method == "llm_cost":
        prefix = "llm_cost"
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



def save_decision_artifacts(
    experiment_id: str,
    prompt: str,
    response_text: str,
    memory_mode: str,
    config_id: str,
    config: dict,
    reason: str,
    experiment: str,
    predicted_cost: str = "",
    cost_reason: str = "",
) -> None:
    """Save the exact LLM decision context for reproducibility."""

    decision_directory = (
        LLM_LOG_DIR / experiment_id
    )

    decision_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    (decision_directory / "prompt.txt").write_text(
        prompt,
        encoding="utf-8",
    )

    (decision_directory / "raw_response.txt").write_text(
        response_text,
        encoding="utf-8",
    )

    decision = {
        "experiment_id": experiment_id,
        "model": LLM_MODEL,
        "memory_mode": memory_mode,
        "configuration_id": config_id,
        "configuration": config,
        "reason": reason,
        "predicted_cost": predicted_cost,
        "cost_reason": cost_reason,
        "experiment": experiment,
        "results_path": str(RESULTS_PATH),
    }

    (decision_directory / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Saved LLM decision artifacts to "
        f"{decision_directory}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and validate the next "
            "LLM-guided search proposal."
        )
    )

    parser.add_argument(
        "--memory-mode",
        choices=[
            "metrics",
            "summary",
            "cost",
        ],
        default="metrics",
    )

    parser.add_argument(
        "--call-api",
        action="store_true",
        help=(
            "Call the fixed study LLM directly "
            "using the configured environment."
        ),
    )

    parser.add_argument(
        "--response-file",
        type=Path,
        help=(
            "Read an offline JSON response instead "
            "of calling the LLM API."
        ),
    )

    parser.add_argument(
        "--save-prompt",
        type=Path,
    )

    args = parser.parse_args()

    if (
        args.call_api
        and args.response_file is not None
    ):
        parser.error(
            "Use either --call-api or "
            "--response-file, not both."
        )

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

    if args.call_api:
        print(
            f"Calling fixed LLM model: "
            f"{LLM_MODEL}"
        )

        response_text = query_llm(
            prompt
        )

    elif args.response_file is not None:
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

    else:
        print(prompt)
        print(
            f"\nFixed study model: {LLM_MODEL}"
        )
        print(
            "\nNo experiment was added. "
            "Use --call-api for a live selection "
            "or --response-file for offline testing."
        )
        return

    (
        config_id,
        reason,
        predicted_cost,
        cost_reason,
    ) = parse_response(
        response_text,
        args.memory_mode,
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


    save_decision_artifacts(
        experiment_id=row["experiment_id"],
        prompt=prompt,
        response_text=response_text,
        memory_mode=args.memory_mode,
        config_id=config_id,
        config=config,
        reason=reason,
        experiment=row["experiment"],
        predicted_cost=predicted_cost,
        cost_reason=cost_reason,
    )

    rows.append(row)
    save_results(rows)

    print(f"Model: {LLM_MODEL}")
    print(f"Selected: {config_id}")
    print(f"Configuration: {config}")
    print(f"Reason: {reason}")
    print(f"Experiment: {row['experiment']}")
    print(f"Added as: {row['experiment_id']}")


if __name__ == "__main__":
    main()
