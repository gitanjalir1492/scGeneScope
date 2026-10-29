#!/usr/bin/env bash
# One-off: create a Python 3.11 venv on shared storage with the scGeneScope dependencies.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/config.sh"

mkdir -p "$OUT_ROOT"
uv venv --python 3.11 "$VENV"

REQS="$OUT_ROOT/requirements.lock.txt"
cd "$REPO"
if uvx --python 3.11 --from 'poetry==1.8.5' poetry export \
        --without-hashes -f requirements.txt -o "$REQS"; then
    cd "$OUT_ROOT"
    # --no-deps: the exported lock has a transitive widgetsnbextension conflict uv refuses to resolve.
    uv pip install --python "$PYTHON" --no-deps -r "$REQS"
    # Fills in deps added to pyproject.toml after the lock was generated, keeping existing pins.
    uv pip install --python "$PYTHON" -e "$REPO"
else
    echo "poetry export failed (lock file out of date?); installing unpinned from pyproject.toml" >&2
    uv pip install --python "$PYTHON" -e "$REPO"
fi

"$PYTHON" -c "import torch, lightning, scgenescope; print('torch', torch.__version__, 'lightning', lightning.__version__)"
