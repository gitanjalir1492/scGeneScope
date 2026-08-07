import argparse
import csv
import json
from pathlib import Path

from llm_search import LLM_MODEL
from results_io import atomic_write_csv


SEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SEARCH_DIR / "results" / "master_results.csv"

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
        raise FileNotFoundError(
            f"Results file does not exist: {RESULTS_PATH}"
        )

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
    atomic_write_csv(
        path=RESULTS_PATH,
        fieldnames=FIELDS,
        rows=rows,
    )


def find_experiment(rows, experiment_id):
    matches = [
        row
        for row in rows
        if row.get("experiment_id") == experiment_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one experiment with ID "
            f"'{experiment_id}', found {len(matches)}."
        )

    return matches[0]


def validate_target(row):
    if row.get("search_method") != "llm_summary":
        raise ValueError(
            "Only llm_summary experiments should receive "
            "automatic experiment summaries."
        )

    if row.get("status") != "completed":
        raise ValueError(
            "Experiment must be completed before summarization."
        )

    if not row.get("val_accuracy", "").strip():
        raise ValueError(
            "Completed experiment has no validation accuracy."
        )

    if not row.get("val_f1", "").strip():
        raise ValueError(
            "Completed experiment has no validation F1."
        )


def format_configuration(row):
    return (
        f"Model setting: {row.get('model_setting')}\n"
        f"Profile setting: {row.get('profile_setting')}\n"
        f"RNA encoder: {row.get('rna_encoder') or None}\n"
        f"Image encoder: {row.get('image_encoder') or None}\n"
        f"Aggregation: {row.get('aggregation') or None}\n"
        f"Fusion: {row.get('fusion') or None}"
    )


def format_previous_history(rows, current_id):
    previous = [
        row
        for row in rows
        if (
            row.get("search_method") == "llm_summary"
            and row.get("status") == "completed"
            and row.get("experiment_id") != current_id
        )
    ]

    if not previous:
        return "No previous llm_summary experiments have been completed."

    blocks = []

    for row in previous:
        summary = (
            row.get("experiment_summary")
            or "not available"
        )

        block = (
            f"Experiment ID: {row.get('experiment_id')}\n"
            f"{format_configuration(row)}\n"
            f"Validation accuracy: {row.get('val_accuracy')}\n"
            f"Validation F1: {row.get('val_f1')}\n"
            f"Experiment summary: {summary}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(rows, row):
    history = format_previous_history(
        rows,
        row["experiment_id"],
    )

    selection_reason = (
        row.get("selection_reason", "").strip()
        or "No selection reason was recorded."
    )

    return f"""
You are maintaining scientific memory for a sequential ML model-search
study in multimodal cellular profiling.

Summarize the completed experiment below so that another ML research
agent can use the summary when choosing future experiments.

The summary should explain:
1. what was tested,
2. what the validation result suggests,
3. what remains uncertain.

Important:
- Base conclusions only on the information provided.
- Validation accuracy is the primary metric.
- Validation F1 is supporting evidence.
- Do not mention held-out test performance.
- Do not claim that one design choice caused a performance difference
  unless the available evidence isolates that design choice.
- If there is insufficient comparative evidence, describe the result as
  an initial observation or baseline.
- Preserve uncertainty.
- Keep the summary to approximately 2 to 4 sentences.

Current experiment:
Experiment ID: {row['experiment_id']}

Configuration:
{format_configuration(row)}

Reason selected:
{selection_reason}

Validation accuracy:
{row['val_accuracy']}

Validation F1:
{row['val_f1']}

Previous completed llm_summary experiments:
{history}

Return JSON only in exactly this format:
{{
  "summary": "Concise scientific summary."
}}
""".strip()


def query_llm(prompt):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The OpenAI SDK is not installed."
        ) from error

    client = OpenAI()

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
        max_output_tokens=400,
    )

    text = response.output_text

    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            "The LLM returned an empty summary."
        )

    return text.strip()


def clean_json_text(text):
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


def parse_summary(text):
    cleaned = clean_json_text(text)

    try:
        response = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(
            "LLM summary response is not valid JSON."
        ) from error

    if not isinstance(response, dict):
        raise ValueError(
            "LLM summary response must be a JSON object."
        )

    summary = response.get("summary")

    if not isinstance(summary, str):
        raise ValueError(
            "LLM response is missing a string summary."
        )

    summary = summary.strip()

    if not summary:
        raise ValueError(
            "Experiment summary cannot be empty."
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate scientific memory for a completed "
            "llm_summary experiment."
        )
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the prompt without calling the API "
            "or modifying master_results.csv."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of an existing summary."
        ),
    )

    args = parser.parse_args()

    rows = load_results()

    row = find_experiment(
        rows,
        args.experiment_id,
    )

    validate_target(row)

    existing_summary = (
        row.get("experiment_summary", "").strip()
    )

    if existing_summary and not args.overwrite:
        raise ValueError(
            "Experiment already has a summary. "
            "Use --overwrite to replace it."
        )

    prompt = build_prompt(
        rows,
        row,
    )

    if args.dry_run:
        print(prompt)
        print(
            "\nDry run complete. No API call was made "
            "and master_results.csv was not modified."
        )
        return

    print(
        f"Calling fixed LLM model: {LLM_MODEL}"
    )

    response_text = query_llm(
        prompt
    )

    summary = parse_summary(
        response_text
    )

    row["experiment_summary"] = summary

    save_results(rows)

    print(
        f"Experiment: {row['experiment_id']}"
    )

    print(
        f"Summary: {summary}"
    )

    print(
        f"\nUpdated results file:\n{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
