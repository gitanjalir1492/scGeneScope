#!/bin/bash
#SBATCH --job-name=scgs_llm_replay
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# One array task per (model, prompting condition); see submit.sh.
set -euo pipefail

: "${REPO:=/novo/users/vnij/projects/scGeneScope}"
: "${VENV:=/novo/projects/departments/mi/vnij/scgenescope/venv}"
: "${REPLICATES:=50}"
: "${BUDGET:=10}"
read -r -a MODELS <<< "${LLM_MODELS:-openai_gpt6_astra anthropic_claude_opus_5_5}"
read -r -a METHODS <<< "${LLM_METHODS:-llm_metrics llm_summary llm_metrics_shuffled llm_metrics_budget llm_metrics_no_isolation llm_metrics_combined llm_metrics_combined_gpu}"

task=${SLURM_ARRAY_TASK_ID:?}
model=${MODELS[$((task / ${#METHODS[@]}))]}
method=${METHODS[$((task % ${#METHODS[@]}))]}

OPENAI_API_KEY=$(< "$HOME/openai_api_key.txt")
export OPENAI_API_KEY
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.marketplace.novo-genai.com/v1}
export PYTHONUNBUFFERED=1

echo "$(date -Is) model=$model method=$method replicates=$REPLICATES budget=$BUDGET host=$(hostname)"
cd "$REPO/experiments/search"
"$VENV/bin/python" replay_search.py --methods "$method" --llm-model "$model" \
    --replicates "$REPLICATES" --budget "$BUDGET"
echo "$(date -Is) done"
