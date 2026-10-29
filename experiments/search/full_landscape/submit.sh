#!/usr/bin/env bash
# Submit the full-fidelity landscape as a SLURM array.
#   ./submit.sh            -> all 51 configs (completed ones exit immediately)
#   ./submit.sh 3,7,40-45  -> only selected task indices (e.g. reruns)
set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export SCGS_CONFIG="$SCRIPT_DIR/config.sh"
source "$SCGS_CONFIG"

ARRAY="${1:-1-51}"

[[ -x "$PYTHON" ]] || { echo "Missing venv at $VENV; run setup_env.sh first." >&2; exit 1; }
# DEPENDENCY (e.g. afterok:<download job id>) lets the array wait for the data download.
if [[ -z "${DEPENDENCY:-}" ]]; then
    [[ -d "$DATA_DIR/features" ]] || { echo "Missing $DATA_DIR/features; run download_data.sh first." >&2; exit 1; }
fi

mkdir -p "$SLURM_LOG_DIR" "$RUNS_DIR" "$LOGS_DIR"
"$PYTHON" "$SCRIPT_DIR/create_manifest.py" --out-root "$OUT_ROOT"

sbatch \
    ${DEPENDENCY:+--dependency="$DEPENDENCY"} \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --array="${ARRAY}%${MAX_CONCURRENT}" \
    --output="$SLURM_LOG_DIR/%x_%A_%a.out" \
    --export=ALL,SCGS_CONFIG="$SCGS_CONFIG" \
    "$SCRIPT_DIR/run_task.sbatch"

echo "Monitor: squeue -u $USER -n scgs_full"
echo "Collect: $PYTHON $SCRIPT_DIR/collect_results.py --out-root $OUT_ROOT"
