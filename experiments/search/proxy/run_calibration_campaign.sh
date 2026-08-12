#!/bin/bash
set -e

cd ~/scGeneScope

echo "Waiting for random_010 proxy to finish..."

while pgrep -f "run_proxy_calibration.py --experiment-id random_010" > /dev/null
do
    sleep 60
done

echo "random_010 proxy is no longer running."
echo "Starting remaining proxy calibrations."

for ID in \
    random_001 \
    random_002 \
    random_003 \
    random_004 \
    random_005 \
    random_006 \
    random_007
do
    echo ""
    echo "========================================"
    echo "Starting $ID"
    echo "========================================"

    poetry run python \
        experiments/search/proxy/run_calibration.py \
        --experiment-id "$ID" \
        --gpu 0

    echo "Finished $ID"
done

echo ""
echo "All remaining proxy calibrations finished."

poetry run python experiments/search/proxy/analyze_calibration.py
