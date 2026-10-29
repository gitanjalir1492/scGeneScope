"""Merge per-config row files into full_landscape_results.csv.

Uses the same columns as the proxy results files, plus training-length and
SLURM metadata, so it can be joined with proxy_landscape_results.csv on `experiment`.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from results_io import atomic_write_csv  # noqa: E402
from run_experiments import FIELDNAMES  # noqa: E402

EXTRA_FIELDS = [
    "epochs_trained",
    "best_epoch",
    "max_epochs",
    "hit_max_epochs",
    "seed",
    "runtime_hours",
    "best_checkpoint",
    "slurm_job_id",
    "node",
    "gpu_name",
    "restart_count",
]


def training_length(run_dir: Path) -> dict[str, str]:
    info = {}
    metrics_files = sorted(run_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime)
    if metrics_files:
        with metrics_files[-1].open(newline="", encoding="utf-8") as f:
            val_rows = [
                r for r in csv.DictReader(f)
                if r.get("val/acc") and r.get("epoch")
            ]
        if val_rows:
            best = max(val_rows, key=lambda r: float(r["val/acc"]))
            info["best_epoch"] = str(int(float(best["epoch"])))
            info["epochs_trained"] = str(
                max(int(float(r["epoch"])) for r in val_rows) + 1
            )

    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if hydra_cfg.exists():
        from omegaconf import OmegaConf

        max_epochs = OmegaConf.load(hydra_cfg).trainer.max_epochs
        info["seed"] = str(OmegaConf.load(hydra_cfg).get("seed"))
        info["max_epochs"] = str(max_epochs)
        if "epochs_trained" in info:
            info["hit_max_epochs"] = str(int(info["epochs_trained"]) >= max_epochs)

    ckpts = sorted((run_dir / "checkpoints").glob("epoch=*.ckpt"))
    if ckpts:
        info["best_checkpoint"] = str(ckpts[-1])
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    rows_dir = args.out_root / "results" / "rows"
    with (args.out_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        ids = [r["experiment_id"] for r in csv.DictReader(f)]

    merged = []
    for exp_id in ids:
        with (rows_dir / f"{exp_id}.csv").open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        row.update({k: "" for k in EXTRA_FIELDS})

        sidecar = rows_dir / f"{exp_id}.slurm.json"
        if sidecar.exists():
            row.update({k: str(v) for k, v in json.loads(sidecar.read_text()).items()})

        if row["started_at"] and row["finished_at"]:
            seconds = (
                datetime.fromisoformat(row["finished_at"])
                - datetime.fromisoformat(row["started_at"])
            ).total_seconds()
            row["runtime_hours"] = f"{seconds / 3600:.3f}"

        if row["run_directory"] and Path(row["run_directory"]).exists():
            row.update(training_length(Path(row["run_directory"])))

        merged.append(row)

    output = args.out_root / "results" / "full_landscape_results.csv"
    atomic_write_csv(path=output, fieldnames=FIELDNAMES + EXTRA_FIELDS, rows=merged)

    print(f"Wrote {output}")
    print("Status:", dict(Counter(r["status"] for r in merged)))
    hit_cap = [r["experiment_id"] for r in merged if r["hit_max_epochs"] == "True"]
    print(f"Runs that hit max_epochs: {hit_cap or 'none'}")
    hours = sum(float(r["runtime_hours"]) for r in merged if r["runtime_hours"])
    print(f"Total GPU-hours so far: {hours:.1f}")


if __name__ == "__main__":
    main()
