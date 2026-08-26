#!/usr/bin/env bash

set -u

RESULTS_FILE="$1"
GPU="$2"

echo "Landscape shard starting"
echo "Results: $RESULTS_FILE"
echo "Physical GPU: $GPU"
echo

while true; do
    NEXT_ID=$(
        python - "$RESULTS_FILE" <<'PY'
import csv
import sys

path = sys.argv[1]

with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("status") == "planned":
            print(row["experiment_id"])
            break
PY
    )

    if [ -z "$NEXT_ID" ]; then
        echo "No planned experiments remain."
        break
    fi

    echo
    echo "========================================"
    echo "Running $NEXT_ID on physical GPU $GPU"
    echo "========================================"

    CUDA_VISIBLE_DEVICES="$GPU" \
    SCGENESCOPE_RESULTS_PATH="$RESULTS_FILE" \
    /homes/55/gitanjali/.cache/pypoetry/virtualenvs/scgenescope-we2uPk9p-py3.11/bin/python \
    experiments/search/run_experiments.py \
        --experiment-id "$NEXT_ID" \
        --execute \
        --proxy

    RC=$?

    if [ "$RC" -ne 0 ]; then
        echo "$NEXT_ID failed with runner exit code $RC."
        echo "Continuing to next planned experiment."
    fi
done

echo
echo "Landscape shard finished."
