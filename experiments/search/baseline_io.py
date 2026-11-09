"""Settings-derived storage and validation for classical replay trajectories."""

from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

import classical_search
from results_io import atomic_write_csv
from search_space import build_experiment_name


def protocol(search_dir, landscape_path, landscape, budget, startup, seed, functions=()):
    sources = {}
    for name in ("classical_search.py", "tpe_acquisition.py", "bayesian_search.py", "random_search.py", "search_space.py"):
        sources[name] = hashlib.sha256((search_dir / name).read_bytes()).hexdigest()
    for function in functions:
        sources[function.__name__] = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    metadata = {
        "schema_version": 1, "protocol": "natural", "landscape": landscape,
        "landscape_sha256": hashlib.sha256(landscape_path.read_bytes()).hexdigest(),
        "budget": budget, "startup_trials": startup, "base_seed": seed,
        "objective": "val_accuracy", "feedback_decimals": 6,
        "sources": sources, "python": platform.python_version(),
        "packages": {name: version(name) for name in ("optuna", "numpy", "scipy", "scikit-learn")},
        "maximum_tpe_proposals": classical_search.MAX_PROPOSALS,
        "rf_trees": classical_search.RF_TREES,
        "startup_not_applicable": ["random", "stratified_random", "multimodal_random", "multimodal_diverse"],
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata["protocol_hash"] = digest
    name = f"budget_{budget:03d}_startup_{startup:03d}_seed_{seed}_p{digest[:12]}"
    return Path(landscape) / name, metadata


@contextmanager
def writer(directory, metadata):
    """One writer per condition; completed CSVs are replaced atomically."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another replay is writing {directory}") from error
        path = directory / "metadata.json"
        if path.exists():
            if json.loads(path.read_text()) != metadata:
                raise ValueError(f"Incompatible metadata in {directory}; refusing to overwrite.")
        else:
            if any(directory.glob("*.csv")):
                raise ValueError("Existing baseline CSVs have no metadata; refusing to adopt them.")
            # Under the directory lock; an interrupted metadata write fails closed next time.
            with path.open("x", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2, sort_keys=True)
                file.write("\n")
        yield


def validate_trials(table, method, budget, landscape, base_seed=None, feedback=None):
    required = {"method", "replicate", "seed", "trial", "experiment", "val_accuracy", "val_f1", "landscape"}
    if not required <= set(table):
        raise ValueError(f"Missing trajectory columns: {sorted(required - set(table))}")
    table = table.copy()
    if table.empty:
        raise ValueError("Empty saved trajectory file.")
    if not table["method"].eq(method).all() or not table["landscape"].eq(landscape).all():
        raise ValueError("Saved method/landscape differs from requested settings.")
    for column in ("replicate", "seed", "trial"):
        values = pd.to_numeric(table[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or (values != np.floor(values)).any() or (values < 0).any():
            raise ValueError(f"Invalid integer trajectory column: {column}")
        table[column] = values.astype(int)
    eligible = {build_experiment_name(c) for c in classical_search.catalogue(method)}
    if not set(table["experiment"]) <= eligible:
        raise ValueError("Saved configuration outside eligible catalogue.")
    for column in ("val_accuracy", "val_f1"):
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column]).all() or not table[column].between(0, 1).all():
            raise ValueError(f"Invalid observed metric: {column}")
    for replicate, sub in table.groupby("replicate"):
        if sorted(sub["trial"].tolist()) != list(range(1, budget + 1)) or sub["experiment"].duplicated().any():
            raise ValueError(f"Malformed/incomplete replicate {replicate}; preserving file for inspection.")
        if sub["seed"].nunique() != 1 or (base_seed is not None and not sub["seed"].eq(base_seed + replicate).all()):
            raise ValueError(f"Seed mismatch in replicate {replicate}.")
    if feedback is not None:
        for row in table.to_dict("records"):
            for metric in ("val_accuracy", "val_f1"):
                expected = float(f"{feedback[row['experiment']][metric]:.6f}")
                if abs(row[metric] - expected) > 1e-12:
                    raise ValueError("Saved observation differs from evaluator feedback.")
    return table


def run_method(directory, metadata, method, replicates, feedback, replay):
    """Resume whole replicates only. Never filter rows before writing them back."""
    with writer(directory, metadata):
        path = directory / f"{method}.csv"
        records = []
        if path.exists():
            saved = validate_trials(pd.read_csv(path, keep_default_na=False), method, metadata["budget"],
                                    metadata["landscape"], metadata["base_seed"], feedback)
            records = saved.to_dict("records")
        done = {row["replicate"] for row in records}
        if done:
            print(f"{method}: resuming, {len(done)} complete replicate(s) saved")
        for replicate in range(replicates):
            if replicate in done:
                continue
            new = replay(method, replicate, feedback, metadata["budget"], metadata["startup_trials"],
                         base_seed=metadata["base_seed"])
            new = [dict(row, landscape=metadata["landscape"]) for row in new]
            validate_trials(pd.DataFrame(new), method, metadata["budget"], metadata["landscape"],
                            metadata["base_seed"], feedback)
            records.extend(new)
            fields = list(dict.fromkeys(name for row in records for name in row))
            atomic_write_csv(path, fields, records)
            print(f"{method}: replicate {replicate + 1}/{replicates} done")
