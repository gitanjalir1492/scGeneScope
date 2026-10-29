# Shared settings for the full-fidelity landscape. Any value can be overridden via the environment.

: "${OUT_ROOT:=/novo/projects/departments/mi/vnij/scgenescope}"
: "${REPO:=/novo/users/vnij/projects/scGeneScope}"
: "${VENV:=$OUT_ROOT/venv}"
# train.py resolves paths.data_dir as $PROJECT_ROOT/data, and the jobs set PROJECT_ROOT=$OUT_ROOT.
DATA_DIR="$OUT_ROOT/data"

: "${PARTITION:=g6e}"          # 1x L40S, 16 CPUs, ~240 GB RAM; pca/resnet50 h5ad files are 40+ GB each
: "${GRES:=gpu:1}"
: "${CPUS:=16}"
: "${MEM:=200G}"
: "${TIME:=2-00:00:00}"
: "${MAX_CONCURRENT:=10}"
: "${SEED:=42}"

: "${HF_TOKEN_FILE:=$HOME/hf_read_access_token_vnij.txt}"

PYTHON="$VENV/bin/python"
SEARCH_DIR="$REPO/experiments/search"
SCRIPT_DIR="$SEARCH_DIR/full_landscape"
MANIFEST="$OUT_ROOT/manifest.csv"
ROWS_DIR="$OUT_ROOT/results/rows"
RUNS_DIR="$OUT_ROOT/runs"
LOGS_DIR="$OUT_ROOT/logs"
SLURM_LOG_DIR="$OUT_ROOT/slurm_logs"
