#!/usr/bin/env bash
# Evaluate the saved checkpoints of the full landscape on the held-out test set.
#   ./submit_test.sh            -> all 51 configs (already evaluated ones exit immediately)
#   ./submit_test.sh 3,7,40-45  -> only selected task indices
#   TEST_DEVICE=cpu PARTITION=highmem ./submit_test.sh  -> CPU inference (models run on precomputed features)
set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export SCGS_CONFIG="$SCRIPT_DIR/config.sh"
source "$SCGS_CONFIG"

ARRAY="${1:-1-51}"
mkdir -p "$SLURM_LOG_DIR" "$OUT_ROOT/test_runs" "$OUT_ROOT/results/test_rows"

GRES_ARG=("--gres=$GRES")
if [[ "${TEST_DEVICE:-gpu}" == "cpu" ]]; then
    GRES_ARG=()
    export TEST_EXTRA_ARGS="trainer.accelerator=cpu"
fi

sbatch \
    --partition="$PARTITION" \
    "${GRES_ARG[@]}" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="${TEST_TIME:-06:00:00}" \
    --array="${ARRAY}%${MAX_CONCURRENT}" \
    --output="$SLURM_LOG_DIR/%x_%A_%a.out" \
    --export=ALL,SCGS_CONFIG="$SCGS_CONFIG" \
    "$SCRIPT_DIR/run_test.sbatch"

echo "Monitor: squeue -u $USER -n scgs_test"
echo "Collect: $PYTHON $SCRIPT_DIR/collect_results.py --out-root $OUT_ROOT"
