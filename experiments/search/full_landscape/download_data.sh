#!/usr/bin/env bash
#SBATCH --job-name=scgs_download
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# Downloads the ~192 GB of embeddings used by the 51 configurations.
# Run directly or with: sbatch --output=<dir>/%x_%j.out download_data.sh
set -euo pipefail
source "${SCGS_CONFIG:-$(dirname "$(readlink -f "$0")")/config.sh}"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
    export HF_TOKEN
fi
export HF_HUB_ENABLE_HF_TRANSFER=1

FILES=()
for emb in rnaseq/pca/n2000 rnaseq/scvi/n200 rnaseq/scgpt imaging/imagenet/vit-l imaging/imagenet/resnet50; do
    for r in round_1 round_2; do
        FILES+=("features/$emb/$r.h5ad")
    done
done

mkdir -p "$DATA_DIR"
"$VENV/bin/huggingface-cli" download altoslabs/scGeneScope "${FILES[@]}" \
    --repo-type dataset --local-dir "$DATA_DIR"

du -sh "$DATA_DIR/features"
