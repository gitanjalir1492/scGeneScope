import csv
from pathlib import Path

from search_space import (
    generate_all_configurations,
    build_experiment_name,
)

OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "proxy_landscape_results.csv"
)

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

configs = generate_all_configurations(
    runnable_only=True
)

if len(configs) != 51:
    raise RuntimeError(
        f"Expected 51 runnable configs, got {len(configs)}"
    )

rows = []

for i, config in enumerate(configs, start=1):
    rows.append(
        {
            "experiment_id": f"landscape_{i:03d}",
            "search_method": "landscape",
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
            "selection_reason": (
                "Exhaustive proxy landscape evaluation"
            ),
            "experiment_summary": "",
            "started_at": "",
            "finished_at": "",
            "return_code": "",
            "run_directory": "",
            "log_file": "",
            "error_message": "",
        }
    )

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
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

print(f"Created: {OUTPUT}")
print(f"Configurations: {len(rows)}")
