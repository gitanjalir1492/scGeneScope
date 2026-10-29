"""Write the 51-config manifest plus one single-row results CSV per config.

IDs follow the proxy landscape order, so full_NNN matches landscape_NNN.
Each SLURM task owns its own row file, avoiding concurrent writes to a shared CSV.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from results_io import atomic_write_csv  # noqa: E402
from run_experiments import FIELDNAMES  # noqa: E402
from search_space import (  # noqa: E402
    build_experiment_name,
    generate_all_configurations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    configs = generate_all_configurations(runnable_only=True)
    if len(configs) != 51:
        raise RuntimeError(f"Expected 51 runnable configs, got {len(configs)}")

    rows_dir = args.out_root / "results" / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    created = 0
    for i, config in enumerate(configs, start=1):
        row = {name: "" for name in FIELDNAMES}
        row.update(
            experiment_id=f"full_{i:03d}",
            search_method="full_landscape",
            model_setting=config["model_setting"],
            profile_setting=config["profile_setting"],
            rna_encoder=config["rna_encoder"] or "",
            image_encoder=config["image_encoder"] or "",
            aggregation=config["aggregation"] or "",
            fusion=config["fusion"] or "",
            experiment=build_experiment_name(config),
            status="planned",
            selection_reason="Exhaustive full-fidelity landscape evaluation",
        )
        manifest.append(row)

        row_path = rows_dir / f"{row['experiment_id']}.csv"
        if not row_path.exists():
            atomic_write_csv(path=row_path, fieldnames=FIELDNAMES, rows=[row])
            created += 1

    manifest_path = args.out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Manifest: {manifest_path} ({len(manifest)} configs)")
    print(f"New row files: {created} (existing ones left untouched)")


if __name__ == "__main__":
    main()
