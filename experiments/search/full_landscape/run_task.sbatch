#!/usr/bin/env bash
#SBATCH --job-name=scgs_full
#SBATCH --requeue
# Array task N trains config full_NNN with the full (non-proxy) settings.
# Partition, resources, array range and log path are passed by submit.sh.
set -euo pipefail
source "$SCGS_CONFIG"

ID=$(printf "full_%03d" "$SLURM_ARRAY_TASK_ID")
ROW="$ROWS_DIR/$ID.csv"

echo "Task $ID | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Skip finished configs; reset failed/interrupted ones so the runner accepts them.
STATUS=$("$PYTHON" - "$ROW" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
row = rows[0]
if row["status"] != "completed":
    for key in ("status", "val_accuracy", "val_f1", "started_at", "finished_at",
                "return_code", "run_directory", "log_file", "error_message"):
        row[key] = ""
    row["status"] = "planned"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
print(row["status"])
PY
)

if [[ "$STATUS" == "completed" ]]; then
    echo "$ID already completed; nothing to do."
    exit 0
fi

"$PYTHON" - "$OUT_ROOT/results/rows/$ID.slurm.json" <<'PY'
import json, os, socket, subprocess, sys
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    gpu = ""
json.dump({
    "slurm_job_id": f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}",
    "node": socket.gethostname(),
    "gpu_name": gpu,
    "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
}, open(sys.argv[1], "w"), indent=2)
PY

export PROJECT_ROOT="$OUT_ROOT"
export SCGENESCOPE_RESULTS_PATH="$ROW"
export SCGENESCOPE_RUN_DIR="$RUNS_DIR"
export SCGENESCOPE_LOG_DIR="$LOGS_DIR"
export HYDRA_FULL_ERROR=1
# Required by cuBLAS when deterministic algorithms are enabled.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$REPO"
"$PYTHON" "$SEARCH_DIR/run_experiments.py" \
    --experiment-id "$ID" \
    --execute \
    --python "$PYTHON" \
    --seed "$SEED"
