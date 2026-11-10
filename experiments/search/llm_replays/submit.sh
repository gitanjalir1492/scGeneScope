#!/bin/bash
# Submit Astra + Opus lookup replays (7 prompting conditions each) as one SLURM array.
# Resumes from saved replicates in results/lookup_replays/models/<model>/<method>.csv.
#   REPLICATES=50 ./submit.sh
#   SCRIPT=unpaired_history.py ./submit.sh   # unpaired-history control
set -euo pipefail
cd "$(dirname "$0")"

: "${PARTITION:=compute}"
: "${TIME:=3-00:00:00}"
: "${SLURM_LOG_DIR:=/novo/projects/departments/mi/vnij/scgenescope/slurm_logs/llm_replays}"
read -r -a MODELS <<< "${LLM_MODELS:-openai_gpt6_astra anthropic_claude_opus_5_5}"
read -r -a METHODS <<< "${LLM_METHODS:-llm_metrics llm_summary llm_metrics_shuffled llm_metrics_budget llm_metrics_no_isolation llm_metrics_combined llm_metrics_combined_gpu}"
mkdir -p "$SLURM_LOG_DIR"

n=$(( ${#MODELS[@]} * ${#METHODS[@]} ))
sbatch --parsable --partition="$PARTITION" --time="$TIME" --array="0-$((n - 1))" \
    --output="$SLURM_LOG_DIR/%A_%a.out" --export=ALL run_llm_replay.sbatch
