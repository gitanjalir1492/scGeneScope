import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SEARCH_DIR / "master_results.csv"

RANDOM_SCRIPT = SEARCH_DIR / "random_search.py"
BAYESIAN_SCRIPT = SEARCH_DIR / "bayesian_search.py"
LLM_SCRIPT = SEARCH_DIR / "llm_search.py"
SUMMARY_SCRIPT = SEARCH_DIR / "summarize_experiment.py"
RUNNER_SCRIPT = SEARCH_DIR / "run_experiments.py"

SUPPORTED_METHODS = [
    "random",
    "bayesian",
    "llm_metrics",
    "llm_summary",
]


def load_results() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file does not exist: {RESULTS_PATH}"
        )

    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def method_rows(
    rows: list[dict[str, str]],
    method: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("search_method") == method
    ]


def completed_rows(
    rows: list[dict[str, str]],
    method: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in method_rows(rows, method)
        if row.get("status") == "completed"
    ]


def planned_rows(
    rows: list[dict[str, str]],
    method: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in method_rows(rows, method)
        if row.get("status") == "planned"
    ]


def failed_rows(
    rows: list[dict[str, str]],
    method: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in method_rows(rows, method)
        if row.get("status") == "failed"
    ]


def proposal_command(
    method: str,
) -> list[str]:
    if method == "random":
        return [
            sys.executable,
            str(RANDOM_SCRIPT),
            "--num-experiments",
            "1",
        ]

    if method == "bayesian":
        return [
            sys.executable,
            str(BAYESIAN_SCRIPT),
        ]

    if method == "llm_metrics":
        return [
            sys.executable,
            str(LLM_SCRIPT),
            "--memory-mode",
            "metrics",
            "--call-api",
        ]

    if method == "llm_summary":
        return [
            sys.executable,
            str(LLM_SCRIPT),
            "--memory-mode",
            "summary",
            "--call-api",
        ]

    raise ValueError(
        f"Unsupported search method: {method}"
    )


def run_command(
    command: list[str],
    env: dict[str, str] | None = None,
) -> None:
    print("\nRunning:")
    print(subprocess.list2cmdline(command))
    print()

    result = subprocess.run(
        command,
        cwd=SEARCH_DIR.parent.parent,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}."
        )


def propose_next(
    method: str,
) -> dict[str, str]:
    before_rows = load_results()

    before_ids = {
        row.get("experiment_id")
        for row in before_rows
    }

    command = proposal_command(
        method
    )

    run_command(
        command
    )

    after_rows = load_results()

    new_rows = [
        row
        for row in after_rows
        if (
            row.get("experiment_id") not in before_ids
            and row.get("search_method") == method
        )
    ]

    if len(new_rows) != 1:
        raise RuntimeError(
            "Expected exactly one new experiment "
            f"for {method}, but found {len(new_rows)}."
        )

    row = new_rows[0]

    if row.get("status") != "planned":
        raise RuntimeError(
            "New experiment was not created "
            "with status='planned'."
        )

    return row


def execute_experiment(
    experiment_id: str,
    gpu: int,
) -> None:
    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    command = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--experiment-id",
        experiment_id,
        "--execute",
    ]

    run_command(
        command,
        env=env,
    )


def summarize_experiment(
    experiment_id: str,
) -> None:
    command = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--experiment-id",
        experiment_id,
    ]

    print(
        f"\nGenerating experiment summary for "
        f"{experiment_id}..."
    )

    run_command(
        command
    )


def print_status(
    method: str,
    budget: int,
) -> None:
    rows = load_results()

    completed = completed_rows(
        rows,
        method,
    )

    planned = planned_rows(
        rows,
        method,
    )

    failed = failed_rows(
        rows,
        method,
    )

    print("\nSearch status:")
    print(f"Method: {method}")
    print(
        f"Completed: {len(completed)} / {budget}"
    )
    print(f"Planned: {len(planned)}")
    print(f"Failed: {len(failed)}")

    if planned:
        print("\nNext planned experiment:")
        print(
            f"ID: {planned[0]['experiment_id']}"
        )
        print(
            f"Experiment: "
            f"{planned[0]['experiment']}"
        )


def dry_run(
    method: str,
    budget: int,
    gpu: int,
) -> None:
    rows = load_results()

    completed = completed_rows(
        rows,
        method,
    )

    planned = planned_rows(
        rows,
        method,
    )

    print("Autonomous search dry run")
    print("=========================")
    print(f"Method: {method}")
    print(f"Target budget: {budget}")
    print(f"GPU: {gpu}")
    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    if len(completed) >= budget:
        print(
            "\nBudget has already been reached. "
            "Nothing would be run."
        )
        return

    if planned:
        row = planned[0]

        print(
            "\nA planned experiment already exists."
        )
        print(
            f"The loop would execute: "
            f"{row['experiment_id']}"
        )
        print(
            f"Experiment: {row['experiment']}"
        )

    else:
        print(
            "\nNo planned experiment exists."
        )
        print(
            "The loop would request one new "
            f"{method} proposal."
        )

    if method == "llm_summary":
        print(
            "\nAfter each successful llm_summary "
            "experiment, the loop would generate "
            "an automatic scientific summary before "
            "requesting the next proposal."
        )

    print(
        "\nAfter each successful experiment, "
        "the loop would reload master_results.csv "
        "and continue until the budget is reached."
    )

    print(
        "\nNo proposer was called and no training "
        "was started."
    )


def run_search(
    method: str,
    budget: int,
    gpu: int,
) -> None:
    print(
        f"Starting autonomous {method} search."
    )

    print(f"Target budget: {budget}")
    print(f"GPU: {gpu}")

    while True:
        rows = load_results()

        completed = completed_rows(
            rows,
            method,
        )

        if len(completed) >= budget:
            print(
                "\nSearch budget reached."
            )

            print_status(
                method,
                budget,
            )

            return

        planned = planned_rows(
            rows,
            method,
        )

        if planned:
            row = planned[0]

            print(
                "\nUsing existing planned "
                "experiment."
            )

        else:
            print(
                "\nNo planned experiment exists. "
                "Requesting the next proposal..."
            )

            row = propose_next(
                method
            )

        experiment_id = row[
            "experiment_id"
        ]

        print(
            f"\nExecuting {experiment_id}"
        )

        print(
            f"Configuration: "
            f"{row['experiment']}"
        )

        execute_experiment(
            experiment_id=experiment_id,
            gpu=gpu,
        )

        rows = load_results()

        updated_rows = [
            candidate
            for candidate in rows
            if candidate.get(
                "experiment_id"
            ) == experiment_id
        ]

        if not updated_rows:
            raise RuntimeError(
                "Experiment disappeared from "
                "master_results.csv."
            )

        updated_row = updated_rows[0]

        if (
            updated_row.get("status")
            != "completed"
        ):
            raise RuntimeError(
                f"{experiment_id} ended with "
                f"status="
                f"'{updated_row.get('status')}'. "
                "Stopping the autonomous loop."
            )

        print(
            f"\n{experiment_id} completed."
        )

        print(
            "Validation accuracy: "
            f"{updated_row.get('val_accuracy')}"
        )

        print(
            "Validation F1: "
            f"{updated_row.get('val_f1')}"
        )

        if method == "llm_summary":
            summarize_experiment(
                experiment_id
            )

            rows = load_results()

            summarized_rows = [
                candidate
                for candidate in rows
                if candidate.get(
                    "experiment_id"
                ) == experiment_id
            ]

            if not summarized_rows:
                raise RuntimeError(
                    "Summarized experiment "
                    "disappeared from results."
                )

            summary = (
                summarized_rows[0]
                .get(
                    "experiment_summary",
                    "",
                )
                .strip()
            )

            if not summary:
                raise RuntimeError(
                    "llm_summary experiment "
                    "completed but no experiment "
                    "summary was stored."
                )

            print(
                f"\nStored summary:\n{summary}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an autonomous sequential "
            "scGeneScope search loop."
        )
    )

    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        required=True,
        help=(
            "Search strategy to run."
        ),
    )

    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help=(
            "Target total number of completed "
            "experiments for this search method."
        ),
    )

    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help=(
            "Physical GPU index exposed to each "
            "training experiment."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually run the autonomous loop. "
            "Without this flag, only preview "
            "what would happen."
        ),
    )

    args = parser.parse_args()

    if args.budget < 1:
        parser.error(
            "--budget must be at least 1."
        )

    if args.gpu < 0:
        parser.error(
            "--gpu must be zero or greater."
        )

    if args.execute:
        run_search(
            method=args.method,
            budget=args.budget,
            gpu=args.gpu,
        )

    else:
        dry_run(
            method=args.method,
            budget=args.budget,
            gpu=args.gpu,
        )


if __name__ == "__main__":
    main()
