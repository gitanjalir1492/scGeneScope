#!/usr/bin/env bash
#SBATCH --job-name=scgs_test
#SBATCH --requeue
# Array task N evaluates the saved best checkpoint of full_NNN on the held-out test set (no training).
# Partition, resources, array range and log path are passed by submit_test.sh.
set -euo pipefail
source "$SCGS_CONFIG"

ID=$(printf "full_%03d" "$SLURM_ARRAY_TASK_ID")
OUT_JSON="$OUT_ROOT/results/test_rows/$ID.json"
if [[ -f "$OUT_JSON" ]]; then
    echo "$ID already evaluated; nothing to do."
    exit 0
fi

read -r EXPERIMENT RUN_DIR < <("$PYTHON" - "$ROWS_DIR/$ID.csv" <<'PY'
import csv, sys
row = next(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
if row["status"] != "completed":
    sys.exit(f"{row['experiment_id']} is {row['status']!r}, not completed")
print(row["experiment"], row["run_directory"])
PY
)
CKPTS=("$RUN_DIR"/checkpoints/epoch=*.ckpt)
[[ ${#CKPTS[@]} -eq 1 && -f "${CKPTS[0]}" ]] || { echo "Expected one checkpoint in $RUN_DIR/checkpoints" >&2; exit 1; }
CKPT="${CKPTS[0]}"
TEST_DIR="$OUT_ROOT/test_runs/$ID"
rm -rf "$TEST_DIR"

echo "Task $ID | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | node $(hostname) | $CKPT"

export PROJECT_ROOT="$OUT_ROOT"
export HYDRA_FULL_ERROR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$REPO"
"$PYTHON" src/scgenescope/scripts/train.py \
    "experiment=$EXPERIMENT" \
    "hydra.run.dir=$TEST_DIR" \
    train=false \
    test=true \
    "ckpt_path='$CKPT'" \
    "seed=$SEED" \
    trainer.deterministic=warn \
    ${TEST_EXTRA_ARGS:-}

mkdir -p "$(dirname "$OUT_JSON")"
"$PYTHON" - "$TEST_DIR" "$CKPT" "$OUT_JSON" <<'PY'
import csv, json, sys
from pathlib import Path
test_dir, ckpt, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
rows = [r for f in test_dir.rglob("metrics.csv") for r in csv.DictReader(open(f, newline=""))
        if r.get("test/acc")]
if len(rows) != 1:
    sys.exit(f"Expected one test row in {test_dir}, found {len(rows)}")
out.write_text(json.dumps({"test_accuracy": rows[0]["test/acc"], "test_f1": rows[0]["test/f1"],
                           "test_loss": rows[0].get("test/loss", ""), "test_checkpoint": ckpt,
                           "test_run_directory": str(test_dir)}, indent=2))
print(out.read_text())
PY
