import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SEARCH_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SEARCH_DIR.parent.parent

MASTER_RESULTS = (
    SEARCH_DIR
    / "results"
    / "master_results.csv"
)

PROXY_RESULTS = (
    SEARCH_DIR
    / "results"
    / "proxy_results.csv"
)

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "scgenescope"
    / "scripts"
    / "train.py"
)

PROXY_RUN_DIR = SEARCH_DIR / "runs" / "proxy"
PROXY_LOG_DIR = SEARCH_DIR / "logs" / "proxy"

PROXY_MAX_EPOCHS = 10
PROXY_MIN_EPOCHS = 1
PROXY_PATIENCE = 2

FIELDNAMES = [
    "proxy_id",
    "source_experiment_id",
    "search_method",
    "model_setting",
    "profile_setting",
    "rna_encoder",
    "image_encoder",
    "aggregation",
    "fusion",
    "experiment",
    "max_epochs",
    "min_epochs",
    "early_stopping_patience",
    "status",
    "val_accuracy",
    "val_f1",
    "started_at",
    "finished_at",
    "return_code",
    "run_directory",
    "log_file",
    "error_message",
]


def read_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def write_proxy_results(rows):
    PROXY_RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = PROXY_RESULTS.with_suffix(
        ".csv.tmp"
    )

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(PROXY_RESULTS)


def parse_float(value):
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def find_metrics_file(run_directory):
    files = list(
        run_directory.rglob("metrics.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No metrics.csv found under "
            f"{run_directory}"
        )

    return max(
        files,
        key=lambda p: p.stat().st_mtime,
    )


def read_validation_metrics(metrics_path):
    rows = read_csv(metrics_path)

    best_accuracy = None
    best_f1 = None
    all_f1 = []

    for row in rows:
        accuracy = parse_float(
            row.get("val/acc")
        )

        f1 = parse_float(
            row.get("val/f1")
        )

        if f1 is not None:
            all_f1.append(f1)

        if accuracy is None:
            continue

        if (
            best_accuracy is None
            or accuracy > best_accuracy
        ):
            best_accuracy = accuracy
            best_f1 = f1

    if best_accuracy is None:
        raise ValueError(
            "No valid val/acc metric found."
        )

    if best_f1 is None and all_f1:
        best_f1 = max(all_f1)

    return best_accuracy, best_f1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a short proxy evaluation "
            "without modifying master_results.csv."
        )
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
        help=(
            "Existing experiment ID from "
            "master_results.csv."
        ),
    )

    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    master_rows = read_csv(
        MASTER_RESULTS
    )

    matches = [
        row
        for row in master_rows
        if row.get("experiment_id")
        == args.experiment_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row for "
            f"{args.experiment_id}, found "
            f"{len(matches)}."
        )

    source = matches[0]

    existing = (
        read_csv(PROXY_RESULTS)
        if PROXY_RESULTS.exists()
        else []
    )

    same_config = [
        row
        for row in existing
        if row.get("source_experiment_id")
        == args.experiment_id
        and row.get("status") == "completed"
    ]

    if same_config:
        raise RuntimeError(
            f"A completed proxy result already "
            f"exists for {args.experiment_id}."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    proxy_id = (
        f"proxy_{args.experiment_id}_"
        f"{timestamp}"
    )

    run_directory = (
        PROXY_RUN_DIR / proxy_id
    ).resolve()

    log_path = (
        PROXY_LOG_DIR
        / f"{proxy_id}.log"
    ).resolve()

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"experiment={source['experiment']}",
        f"hydra.run.dir={run_directory}",
        "test=false",
        f"trainer.max_epochs={PROXY_MAX_EPOCHS}",
        f"trainer.min_epochs={PROXY_MIN_EPOCHS}",
        (
            "callbacks.early_stopping.patience="
            f"{PROXY_PATIENCE}"
        ),
    ]

    proxy_row = {
        "proxy_id": proxy_id,
        "source_experiment_id":
            args.experiment_id,
        "search_method":
            source.get("search_method", ""),
        "model_setting":
            source.get("model_setting", ""),
        "profile_setting":
            source.get("profile_setting", ""),
        "rna_encoder":
            source.get("rna_encoder", ""),
        "image_encoder":
            source.get("image_encoder", ""),
        "aggregation":
            source.get("aggregation", ""),
        "fusion":
            source.get("fusion", ""),
        "experiment":
            source.get("experiment", ""),
        "max_epochs":
            str(PROXY_MAX_EPOCHS),
        "min_epochs":
            str(PROXY_MIN_EPOCHS),
        "early_stopping_patience":
            str(PROXY_PATIENCE),
        "status": "running",
        "val_accuracy": "",
        "val_f1": "",
        "started_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
        "finished_at": "",
        "return_code": "",
        "run_directory":
            str(run_directory),
        "log_file":
            str(log_path),
        "error_message": "",
    }

    existing.append(proxy_row)
    write_proxy_results(existing)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu
    )

    print("\nProxy evaluation")
    print(
        f"Source: {args.experiment_id}"
    )
    print(
        f"Config: {source['experiment']}"
    )
    print(
        f"GPU: {args.gpu}"
    )
    print(
        f"Max epochs: {PROXY_MAX_EPOCHS}"
    )
    print(
        f"Early stopping patience: "
        f"{PROXY_PATIENCE}"
    )
    print("\nCommand:")
    print(" ".join(command))
    print(
        f"\nLog: {log_path}"
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    proxy_row["return_code"] = str(
        result.returncode
    )

    proxy_row["finished_at"] = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    if result.returncode != 0:
        proxy_row["status"] = "failed"
        proxy_row["error_message"] = (
            f"Training returned exit code "
            f"{result.returncode}."
        )

        write_proxy_results(existing)

        raise RuntimeError(
            proxy_row["error_message"]
        )

    metrics_path = find_metrics_file(
        run_directory
    )

    val_accuracy, val_f1 = (
        read_validation_metrics(
            metrics_path
        )
    )

    proxy_row["status"] = "completed"
    proxy_row["val_accuracy"] = str(
        val_accuracy
    )
    proxy_row["val_f1"] = (
        ""
        if val_f1 is None
        else str(val_f1)
    )

    write_proxy_results(existing)

    print("\nProxy completed successfully.")
    print(
        f"Validation accuracy: "
        f"{val_accuracy}"
    )
    print(
        f"Validation F1: {val_f1}"
    )
    print(
        f"Results: {PROXY_RESULTS}"
    )


if __name__ == "__main__":
    main()
