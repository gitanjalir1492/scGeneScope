"""Read-only input loading and offline LLM replay reports. No inference calls."""

from contextlib import contextmanager, redirect_stdout
from copy import copy
import csv
import fcntl
import hashlib
import json
from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd

import baseline_analysis
import baseline_io
import llm_client
import llm_replay_io
import llm_search
import summarize_experiment
from results_io import atomic_write_csv
from search_space import build_experiment_name


def reporting_source_hashes(search_dir):
    """Full reporting sources are audited separately from execution compatibility."""
    return {name: hashlib.sha256((search_dir / name).read_bytes()).hexdigest()
            for name in ("replay_with_report.py", "llm_analysis.py", "baseline_analysis.py")}


def after_replay(args, runner):
    """Report the exact saved condition after the inference writer has released its lock."""
    config = llm_client.settings(args)
    directory = runner.BASE_REPLAY_DIR / "models" / llm_replay_io.readable_path(args, config)
    analysis_args = copy(args)
    analysis_args.analyze_only = True
    analysis_args.analysis_inputs = [directory]
    # Reducing the invocation count must not hide already requested trajectories.
    with reader(directory):
        analysis_args.replicates = max([args.replicates, *[
            llm_replay_io.read_json(directory / "requests" / f"{method}.json")["replicates"]
            for method in args.methods]])
    print("Replay results are saved. Generating the offline summary...")
    return run(analysis_args, runner)


@contextmanager
def reader(directory):
    # Cooperates with baseline_io.writer without creating or changing input files.
    lock = directory / ".writer.lock"
    if not lock.exists():
        raise ValueError(f"Missing protocol lock file in {directory}; not adopting legacy outputs.")
    with lock.open("r") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A replay is still writing {directory}; analyse after it stops.") from error
        yield


def input_spec(spec, global_methods):
    """Parse PATH or PATH::method,method without changing the execution CLI."""
    raw = str(spec)
    path_text, separator, method_text = raw.rpartition("::")
    if not separator:
        return Path(raw).resolve(), tuple(global_methods)
    if not path_text or not method_text:
        raise ValueError(f"Invalid analysis input filter: {raw!r}; use PATH::method,method.")
    methods = tuple(dict.fromkeys(method_text.split(",")))
    unknown = set(methods) - set(global_methods)
    if unknown:
        raise ValueError(f"Input-specific methods must also appear in --methods: {', '.join(sorted(unknown))}.")
    return Path(path_text).resolve(), methods


def observed_cost_prompt_equivalent(saved, rebuilt, exact_total):
    """Allow only the adjacent-cent rendering of an exact half-cent float sum."""
    pattern = re.compile(r"(?m)^GPU hours used so far: ([0-9]+\.[0-9]{2})$")
    old, new = pattern.findall(saved), pattern.findall(rebuilt)
    if (len(old) != 1 or len(new) != 1 or abs(float(old[0]) - float(new[0])) > 0.0100000001
            or abs(float(old[0]) - exact_total) > 0.0050000001
            or abs(float(new[0]) - exact_total) > 0.0050000001):
        return False
    return pattern.sub("GPU hours used so far: <observed-total>", saved) == pattern.sub(
        "GPU hours used so far: <observed-total>", rebuilt)


def validate_saved_method_for_analysis(directory, metadata, method, feedback):
    """Use execution validation, narrowly tolerating historical float display drift."""
    try:
        return llm_replay_io.validate_saved_method(directory, metadata, method, feedback)
    except ValueError as error:
        if str(error) != "Checkpoint decision differs from its observed-history request/answer.":
            raise

    states = {}
    for path in sorted((directory / "checkpoints" / method).glob("*.json")):
        state = llm_replay_io.read_json(path)
        replicate = state.get("replicate")
        llm_replay_io.validate_checkpoint(state, metadata, method, replicate, feedback)
        if path.name != f"replicate_{replicate:03d}.json" or replicate in states:
            raise ValueError(f"Checkpoint filename/identity mismatch: {path}")
        states[replicate] = state
    for log_directory in (directory / "llm_logs" / method).glob("*"):
        if not log_directory.is_dir() or not any(log_directory.name == f"replicate_{r:03d}" for r in states):
            raise ValueError(f"Orphaned inference logs without a checkpoint: {log_directory}")
    tolerated = []
    for replicate, state in states.items():
        accepted = {}
        for path in sorted((directory / "llm_logs" / method / f"replicate_{replicate:03d}").glob("*.json")):
            record = llm_replay_io.read_json(path)
            llm_replay_io.validate_journal(path, record, metadata, method, replicate)
            trial, kind = record["trial"], record["kind"]
            if trial > len(state["rows"]) + (kind == "decision"):
                raise ValueError(f"Journal is ahead of its saved prefix: {path}")
            if record["status"] == "accepted":
                identity = llm_replay_io.read_json(directory / "response_identity.json")
                if identity != llm_replay_io.response_identity(record["response"]):
                    raise ValueError("Accepted response fingerprint differs from the saved server response identity.")
                key = (trial, kind)
                if key in accepted:
                    raise ValueError(f"Multiple accepted responses for one selection: {path}")
                accepted[key] = (path, record)
        for trial, row in enumerate(state["rows"], 1):
            entry = accepted.get((trial, "decision"))
            if entry is None:
                raise ValueError("Checkpoint observation lacks its accepted decision journal.")
            path, record = entry
            variant = metadata["variants"][method]
            prompt, configs, available = llm_search.build_prompt(
                state["rows"][:trial - 1], variant["mode"],
                shuffle_seed=state["seed"] if variant["shuffle"] else None,
                budget=metadata["budget"] if variant["budget"] else None,
                isolation_principles=variant["isolation"], gpu_hours_aware=variant["gpu"])
            saved_prompt = record["request"]["messages"][0]["content"]
            same_prompt = saved_prompt == prompt
            exact_total = sum(float(previous["runtime_hours"]) for previous in state["rows"][:trial - 1])
            if (not same_prompt and variant["gpu"]
                    and observed_cost_prompt_equivalent(saved_prompt, prompt, exact_total)):
                tolerated.append(str(path.relative_to(directory)))
                same_prompt = True
            config_id, reason, _, _ = llm_search.parse_response(
                llm_client.normalize_response(record["response"])["text"], variant["mode"])
            if (not same_prompt or config_id not in available
                    or build_experiment_name(configs[config_id]) != row["experiment"] or reason != row["selection_reason"]):
                raise ValueError("Checkpoint decision differs from its observed-history request/answer.")
            if row["experiment_summary"]:
                summary_entry = accepted.get((trial, "summary"))
                if summary_entry is None:
                    raise ValueError("Checkpoint summary differs from its accepted journal.")
                _, summary = summary_entry
                if (summarize_experiment.parse_summary(llm_client.normalize_response(summary["response"])["text"])
                        != row["experiment_summary"]
                        or summary["request"]["messages"][0]["content"]
                        != summarize_experiment.build_prompt(state["rows"][:trial], row)):
                    raise ValueError("Checkpoint summary differs from its observed history.")
            elif method == "llm_summary" and (trial < len(state["rows"]) or state["status"] == "completed"):
                raise ValueError("Missing summary in a completed checkpoint prefix.")
    if not tolerated:
        raise ValueError("Checkpoint decision differs from its observed-history request/answer.")
    llm_replay_io.check_existing_export(directory, metadata, method, feedback)
    status_path = directory / f"{method}_status.csv"
    if status_path.exists():
        seen = set()
        with status_path.open(newline="") as file:
            for row in csv.DictReader(file):
                replicate = int(row["replicate"])
                state = states.get(replicate)
                if (state is None or replicate in seen or row["method"] != method or int(row["seed"]) != state["seed"]
                        or not 0 <= int(row["n_evaluations"]) <= len(state["rows"])
                        or row["status"] not in {"completed", "failed", "in_progress"}
                        or (row["status"] != "in_progress" and row["status"] != state["status"])):
                    raise ValueError("Saved status CSV differs from its checkpoints; refusing to analyse.")
                seen.add(replicate)
    request_path = directory / "requests" / f"{method}.json"
    count = llm_replay_io.read_json(request_path).get("replicates") if request_path.exists() else 0
    if type(count) is not int or count < 0 or (states and count <= max(states)):
        raise ValueError("Missing or invalid requested-replicate ledger; refusing to lose the failure denominator.")
    print(f"Reporting compatibility: accepted adjacent-cent cumulative observed-cost rendering in {len(tolerated)} "
          f"saved {method} request(s); all individual feedback, request settings, selections and checksums validated.")
    return count


@contextmanager
def analysis_input(directory, methods, args, runner, full_hash):
    """Read modern protocols strictly, or snapshot explicitly selected historical archives."""
    archives = {(runner.BASE_REPLAY_DIR / "models" / model).resolve()
                for model in ("openai_gpt6_astra", "anthropic_claude_opus_5_5")}
    if directory not in archives or (directory / "metadata.json").exists():
        with reader(directory):
            yield json.loads((directory / "metadata.json").read_text()), None
        return
    # This narrow offline adapter cannot adopt a metadata-less local Chat run.
    if any((directory / name).exists() for name in
           (".writer.lock", "checkpoints", "requests", "validation", "response_identity.json")):
        raise ValueError("Historical archive contains modern replay state; refusing a provenance downgrade.")

    def paths():
        return {path for method in methods if method in runner.LLM_VARIANTS
                for path in [directory / f"{method}.csv", *(directory / "llm_logs" / method).glob("*.json")]
                if path.is_file()}

    # Historical writers had no cooperative lock. Use the exact bytes scored and
    # refuse a changing snapshot; never create a lock or metadata in the archive.
    snapshot = {path: path.read_bytes() for path in paths()}
    hashes = {str(path.relative_to(directory)): hashlib.sha256(data).hexdigest()
              for path, data in sorted(snapshot.items())}
    metadata = dict(protocol="historical-saved-csv", protocol_hash=None, model=directory.name,
                    snapshot_sha256=llm_client.digest(dict(model=directory.name, files=hashes)),
                    metadata_origin="offline analysis; not recorded execution metadata",
                    model_identity="directory label; not server verified",
                    reference_landscape_sha256=full_hash, analysis_budget=args.budget,
                    reproducibility_limitations=[
                        "Execution fingerprint, model revision, decoding/reasoning settings and inference seeds were not recorded.",
                        "Estimates and intervals describe saved completed trajectories only; the original failure denominator is unknown.",
                        "Historical text logs do not establish complete request accounting, token usage, latency or cost.",
                        "CSV observations are checked against the current full landscape; the original landscape source hash is unavailable."])
    print(f"Historical archive: {directory.name}; estimates describe saved completed trajectories only. "
          "Execution provenance and failure/inference accounting are unavailable.")
    yield metadata, snapshot
    if paths() != set(snapshot) or any(path.read_bytes() != data for path, data in snapshot.items()):
        raise ValueError(f"Historical inputs changed during analysis: {directory}; stop the writer and rerun offline analysis.")


def request_costs(records, gpu_count=None):
    result = dict(inference_requests=len(records), inference_seconds=0.0, prompt_tokens=0,
                  completion_tokens=0, reasoning_tokens=0, reasoning_chars=0, final_chars=0,
                  invalid_outputs=0, transport_errors=0, pending_requests=0, truncated_outputs=0,
                  missing_usage=0, missing_reasoning_usage=0, missing_latency=0)
    for record in records:
        latency = record.get("latency_seconds")
        result["missing_latency"] += latency is None
        result["inference_seconds"] += latency or 0
        raw = record.get("response")
        raw = raw if isinstance(raw, dict) else {}
        usage = raw.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        measured = lambda value: isinstance(value, int) and not isinstance(value, bool) and value >= 0
        result["missing_usage"] += not all(measured(usage.get(k)) for k in ("prompt_tokens", "completion_tokens"))
        for metric in ("prompt_tokens", "completion_tokens"):
            result[metric] += usage[metric] if measured(usage.get(metric)) else 0
        details = usage.get("completion_tokens_details")
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        result["missing_reasoning_usage"] += not measured(reasoning_tokens)
        result["reasoning_tokens"] += reasoning_tokens if measured(reasoning_tokens) else 0
        choices = raw.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        result["reasoning_chars"] += len(reasoning) if isinstance(reasoning, str) else 0
        result["final_chars"] += len(message["content"]) if isinstance(message.get("content"), str) else 0
        result["invalid_outputs"] += record["status"] == "invalid_output"
        result["transport_errors"] += record["status"] == "transport_error"
        result["pending_requests"] += record["status"] == "pending"
        result["truncated_outputs"] += choice.get("finish_reason") == "length"
    result["inference_request_hours"] = result["inference_seconds"] / 3600
    # A generic endpoint does not establish how many GPUs served a request.
    result["inference_request_gpu_hours_estimate"] = (
        result["inference_request_hours"] * gpu_count if gpu_count is not None else np.nan if records else 0.0)
    return result


def primary_summary(terminal, requested, optimum):
    summaries = []
    for series, sub in terminal.groupby("series", sort=True):
        def measured_total(column):
            return int(sub[column].sum()) if sub[column].notna().all() else np.nan

        n = len(sub)
        n_requested = requested[series]
        finished = sub["status"].isin(["completed", "failed"]).all() and n == n_requested
        started = int(sub["status"].ne("not_started").sum())
        row = dict(series=series, requested_replicates=n_requested, n_started=started,
                   n_completed=int(sub["status"].eq("completed").sum()),
                   n_failed=int(sub["status"].eq("failed").sum()),
                   n_in_progress=int(sub["status"].eq("in_progress").sum()),
                   n_not_started=n_requested - started, is_final=bool(finished),
                   failure_rate=sub["status"].eq("failed").sum() / n_requested if finished else np.nan,
                   failure_rate_observed_so_far=sub["status"].eq("failed").sum() / n_requested,
                   inference_requests=measured_total("inference_requests"),
                   n_with_incumbent=int(sub["best_val_accuracy"].notna().sum()),
                   mean_best_val_accuracy_observed=sub["best_val_accuracy"].mean(),
                   mean_training_gpu_hours_observed=sub["training_gpu_hours"].mean())
        for field in ("api_endpoint", "revision_provenance", "sampling_provenance", "reasoning_control_provenance",
                      "population_scope", "failure_accounting"):
            if field in sub:
                row[field] = sub[field].iloc[0]
        for metric in ("found_best", "found_top3"):
            hits = int(sub[metric].sum())
            row[f"p_{metric}_observed_so_far"] = hits / n_requested
            row[f"p_{metric}_at_budget"] = hits / n if finished else np.nan
            low, high = baseline_analysis.wilson(hits, n)
            row[f"p_{metric}_ci95_low"] = low if finished else np.nan
            row[f"p_{metric}_ci95_high"] = high if finished else np.nan
        missing = sub["best_val_accuracy"].isna().sum()
        row["mean_best_val_accuracy_lower_bound"] = sub["best_val_accuracy"].sum() / n if finished else np.nan
        row["mean_best_val_accuracy_upper_bound"] = (sub["best_val_accuracy"].sum() + missing * optimum) / n if finished else np.nan
        for metric in ("best_val_accuracy", "simple_regret", "training_gpu_hours", "incumbent_test_accuracy", "incumbent_test_f1",
                       "inference_seconds", "inference_request_hours", "inference_request_gpu_hours_estimate", "prompt_tokens", "completion_tokens", "reasoning_tokens"):
            values = (optimum - sub["best_val_accuracy"] if metric == "simple_regret" else sub[metric]).to_numpy(float)
            row[f"mean_{metric}_observed_so_far"] = values.mean()
            row[f"mean_{metric}"] = values.mean() if finished else np.nan
            draws = np.random.default_rng(0).integers(n, size=(baseline_analysis.N_BOOTSTRAP, n))
            low, high = baseline_analysis.intervals(values[:, None], draws)
            row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = (low[0], high[0]) if finished else (np.nan, np.nan)
        for metric in ("invalid_outputs", "transport_errors", "pending_requests", "truncated_outputs",
                       "missing_usage", "missing_reasoning_usage", "missing_latency"):
            row[metric] = measured_total(metric)
        # Missing telemetry must not look like exact zero inference cost.
        for metric, missing_field in (("inference_seconds", "missing_latency"), ("prompt_tokens", "missing_usage"),
                                      ("completion_tokens", "missing_usage"), ("reasoning_tokens", "missing_reasoning_usage"),
                                      ("inference_request_hours", "missing_latency"),
                                      ("inference_request_gpu_hours_estimate", "missing_latency")):
            if pd.notna(row[missing_field]) and row[missing_field]:
                row[f"mean_{metric}_known_lower_bound_observed_so_far"] = row[f"mean_{metric}_observed_so_far"]
                for column in (f"mean_{metric}", f"{metric}_ci95_low", f"{metric}_ci95_high"):
                    row[column] = np.nan
        row["invalid_output_rate_per_request"] = row["invalid_outputs"] / row["inference_requests"] if row["inference_requests"] else np.nan
        row["transport_error_rate_per_request"] = row["transport_errors"] / row["inference_requests"] if row["inference_requests"] else np.nan
        if row.get("failure_accounting") == "unavailable":
            for field in ("n_failed", "n_in_progress", "n_not_started", "failure_rate", "failure_rate_observed_so_far"):
                row[field] = np.nan
        summaries.append(row)
    return pd.DataFrame(summaries)


def all_trajectory_curves(frames, terminal, full, budget):
    """Unconditional curves: failures stop searching; partial studies have no final estimates."""
    if frames:
        scored, optimum, _, _ = baseline_analysis.score_trials(pd.concat(frames, ignore_index=True), full)
    else:
        scored = pd.DataFrame(columns=["series", "replicate", "trial", "cumulative_gpu_hours"])
        optimum = full["val_accuracy"].max()
    maximum_cost = scored["cumulative_gpu_hours"].max() if not scored.empty else 0.0
    grid = np.linspace(0, maximum_cost, baseline_analysis.COMPUTE_GRID_POINTS)
    evaluation_rows, compute_rows = [], []
    for series, endpoints in terminal.groupby("series", sort=True):
        trajectories = {int(r): sub.sort_values("trial") for r, sub in scored.loc[scored["series"].eq(series)].groupby("replicate")}
        n = len(endpoints)
        final = endpoints["status"].isin(["completed", "failed"]).all()
        draws = np.random.default_rng(baseline_analysis.BOOTSTRAP_SEED).integers(n, size=(baseline_analysis.N_BOOTSTRAP, n))
        values = np.full((n, budget), np.nan)
        hits, top3, costs = (np.zeros((n, budget)) for _ in range(3))
        compute_values = np.full((n, len(grid)), np.nan)
        compute_hits = np.zeros_like(compute_values)
        for i, endpoint in enumerate(endpoints.itertuples()):
            trajectory = trajectories.get(endpoint.replicate)
            if trajectory is None or trajectory.empty:
                continue
            # A terminal failure retains its last accepted incumbent and cost.
            indices = np.minimum(np.arange(budget), len(trajectory) - 1)
            for target, column in ((values, "best_so_far"), (hits, "found_best"), (top3, "found_top3"), (costs, "cumulative_gpu_hours")):
                target[i] = trajectory[column].to_numpy(float)[indices]
            compute_values[i], compute_hits[i] = (matrix[0] for matrix in baseline_analysis.at_compute([trajectory], grid))
        for matrices, coordinates, output in (
                ({"best_val_accuracy": values, "simple_regret": optimum - values, "training_gpu_hours": costs,
                  "found_best": hits, "found_top3": top3}, range(1, budget + 1), evaluation_rows),
                ({"best_val_accuracy": compute_values, "found_best": compute_hits}, grid, compute_rows)):
            bounds = {key: baseline_analysis.intervals(matrix, draws) for key, matrix in matrices.items() if not key.startswith("found_")}
            for j, coordinate in enumerate(coordinates):
                row = dict(series=series, n_requested=n, is_final=bool(final),
                           n_with_incumbent=int(np.isfinite(matrices["best_val_accuracy"][:, j]).sum()))
                if "population_scope" in endpoints:
                    row["population_scope"] = endpoints["population_scope"].iloc[0]
                row["trial" if output is evaluation_rows else "cumulative_training_gpu_hours"] = coordinate
                for key, matrix in matrices.items():
                    name = f"p_{key}" if key.startswith("found_") else f"mean_{key}"
                    row[name] = matrix[:, j].mean() if final else np.nan
                    low, high = (baseline_analysis.wilson(matrix[:, j].sum(), n) if key.startswith("found_")
                                 else (bounds[key][0][j], bounds[key][1][j]))
                    row[f"{key}_ci95_low"], row[f"{key}_ci95_high"] = (low, high) if final else (np.nan, np.nan)
                output.append(row)
    return pd.DataFrame(evaluation_rows), pd.DataFrame(compute_rows)


def report_tables(summary, diversity, conditions, budget):
    """Present existing endpoint estimates without recomputing statistical outcomes."""
    identity = pd.DataFrame(conditions).sort_values("series").reset_index(drop=True)
    identity.insert(0, "run", [f"S{i + 1}" for i in range(len(identity))])
    joined = identity.merge(summary, on="series", validate="one_to_one")
    names = dict(requested_replicates="n_replicates", mean_simple_regret="mean_regret",
                 best_val_accuracy_ci95_low="ci95_low", best_val_accuracy_ci95_high="ci95_high",
                 mean_training_gpu_hours="mean_cumulative_gpu_hours",
                 p_found_best_at_budget="p_found_best", p_found_top3_at_budget="p_found_top3")
    metrics = ["requested_replicates", "n_completed", "n_failed", "n_in_progress", "n_not_started", "is_final",
               "mean_best_val_accuracy", "best_val_accuracy_ci95_low", "best_val_accuracy_ci95_high",
               "mean_simple_regret", "p_found_best_at_budget", "p_found_best_ci95_low", "p_found_best_ci95_high",
               "p_found_top3_at_budget", "p_found_top3_ci95_low", "p_found_top3_ci95_high",
               "mean_training_gpu_hours", "mean_incumbent_test_accuracy", "mean_incumbent_test_f1"]
    scope_fields = [name for name in ("population_scope", "failure_accounting") if name in summary]
    metrics += scope_fields
    final = joined[[*identity.columns, *metrics]].rename(columns=names)
    final.insert(len(identity.columns), "trial", budget)
    diagnostic_metrics = ["requested_replicates", "n_completed", "n_failed", "n_in_progress", "n_not_started", "is_final",
                          "inference_requests", "mean_prompt_tokens", "mean_completion_tokens", "mean_reasoning_tokens",
                          "mean_inference_seconds", "invalid_outputs", "transport_errors", "pending_requests",
                          "truncated_outputs", "missing_usage", "missing_reasoning_usage", "missing_latency",
                          "failure_rate", "invalid_output_rate_per_request", "transport_error_rate_per_request"]
    diagnostic_metrics += scope_fields
    diagnostics = joined[[*identity.columns, *diagnostic_metrics]].merge(
        diversity[["series", "n_unique_first", "first_choice_entropy_bits", "first_choice_frequencies",
                   "n_unique_ordered_trajectories", "completed_duplicate_trajectory_rate", "duplicate_ordered_prefix_rate"]],
        on="series", validate="one_to_one")
    return final, diagnostics


def print_report(final, diagnostics):
    def number(value, decimals=4):
        return "NA" if pd.isna(value) else f"{value:.{decimals}f}"

    def interval(row, value, low, high, decimals=4):
        return f"{number(row[value], decimals)} [{number(row[low], decimals)}, {number(row[high], decimals)}]"

    print("LLM replay summary: budget endpoints; brackets are 95% confidence intervals.")
    for row in final.to_dict("records"):
        print(f"{row['run']}: thinking={row['thinking']}, effort={row['reasoning_effort']}, "
              f"profile={row['decoding_profile']}, condition={row['condition']}; input={row['input_directory']}; "
              f"population={row.get('population_scope', 'recorded trajectories')}")
    rows = []
    for row in final.to_dict("records"):
        rows.append(dict(Run=row["run"], Model=row["model"], Prompt=row["method"], Eval=row["trial"], N=row["n_replicates"],
                         **{"Best val [95% CI]": interval(row, "mean_best_val_accuracy", "ci95_low", "ci95_high"),
                            "Regret": number(row["mean_regret"]),
                            "P(opt) [95% CI]": interval(row, "p_found_best", "p_found_best_ci95_low", "p_found_best_ci95_high", 3),
                            "P(top3) [95% CI]": interval(row, "p_found_top3", "p_found_top3_ci95_low", "p_found_top3_ci95_high", 3),
                            "Train GPU-h": number(row["mean_cumulative_gpu_hours"], 2),
                            "Test acc": number(row["mean_incumbent_test_accuracy"]), "Test F1": number(row["mean_incumbent_test_f1"])}))
    print(pd.DataFrame(rows).to_string(index=False))
    rows = []
    for row in diagnostics.to_dict("records"):
        rows.append(dict(Run=row["run"], Completed=f"{row['n_completed']}/{row['requested_replicates']}", Failed=number(row["n_failed"], 0),
                         Unfinished=number(row["n_in_progress"] + row["n_not_started"], 0), Requests=number(row["inference_requests"], 0),
                         **{"In tok/rep": number(row["mean_prompt_tokens"], 0), "Out tok/rep": number(row["mean_completion_tokens"], 0),
                            "Reason tok/rep": number(row["mean_reasoning_tokens"], 0), "Seconds/rep": number(row["mean_inference_seconds"], 1),
                            "Invalid": number(row["invalid_outputs"], 0), "Transport errors": number(row["transport_errors"], 0),
                            "Dup complete %": number(100 * row["completed_duplicate_trajectory_rate"], 1),
                            "Unique first": row["n_unique_first"]}))
    print("Inference and trajectory diagnostics (means per recorded population; modern request journals include all attempts):")
    print(pd.DataFrame(rows).to_string(index=False))
    print("NA denotes unavailable estimates/telemetry, or bootstrap intervals with fewer than two trajectories. "
          "Unfinished conditions have no final estimates; terminal failures remain in the denominator. "
          "Duplicate rate is conditional on completion. Training GPU-hours are separate from inference time.")
    if final.get("input_kind", pd.Series(dtype=str)).eq("historical_csv").any():
        print("Historical rows describe saved completed trajectories only, with intervals across those trajectories. "
              "Their original requested/failure denominator and execution settings are unknown; NA inference costs and "
              "failure counts do not mean zero. Snapshot hashes identify archive contents, not verified execution protocols.")


def run(args, runner):
    full = pd.read_csv(runner.LANDSCAPES["full"])
    feedback = runner.load_landscape("full")
    full_hash = hashlib.sha256(runner.LANDSCAPES["full"].read_bytes()).hexdigest()
    ranked = full.set_index("experiment")["val_accuracy"].sort_values(ascending=False)
    optimum = ranked.iloc[0]
    best, top3 = set(ranked.index[ranked.eq(optimum)]), set(ranked.index[ranked.ge(ranked.iloc[2])])
    full_index = full.set_index("experiment")
    frames, all_frames, terminal, diversity, sources, requested = [], [], [], [], [], {}
    conditions, coverage = [], []
    inputs = [input_spec(spec, args.methods) for spec in args.analysis_inputs]
    if len({directory for directory, _ in inputs}) != len(inputs):
        raise ValueError("Select each analysis input directory once; combine its methods after :: with commas.")
    for directory, input_methods in inputs:
        with analysis_input(directory, input_methods, args, runner, full_hash) as (metadata, archive):
            is_historical = archive is not None
            if not is_historical and (metadata["landscape"] != "full" or metadata["landscape_sha256"] != full_hash or metadata["budget"] != args.budget):
                raise ValueError("Analysis inputs must share the requested full landscape and evaluation budget.")
            if not is_historical and metadata["protocol"] not in {"local-chat-replay", "natural"}:
                raise ValueError("Unsupported analysis protocol; do not adopt unverified historical outputs.")
            is_chat = metadata["protocol"] == "local-chat-replay"
            if is_chat and llm_client.digest({k: v for k, v in metadata.items() if k != "protocol_hash"}) != metadata["protocol_hash"]:
                raise ValueError("Chat metadata fingerprint is invalid.")
            if not is_chat and not is_historical and hashlib.sha256(json.dumps({k: v for k, v in metadata.items() if k != "protocol_hash"}, sort_keys=True).encode()).hexdigest() != metadata["protocol_hash"]:
                raise ValueError("Classical metadata fingerprint is invalid.")
            selected_files = set(archive) if is_historical else {directory / "metadata.json"}
            if is_chat and (directory / "response_identity.json").exists():
                selected_files.add(directory / "response_identity.json")
            for method in input_methods:
                path = directory / f"{method}.csv"
                if is_historical and path not in archive:
                    continue
                checkpoints = directory / "checkpoints" / method
                if not path.exists() and not (is_chat and (any(checkpoints.glob("replicate_*.json")) or (directory / "requests" / f"{method}.json").exists())):
                    continue
                if path.exists():
                    selected_files.add(path)
                if not is_historical and (directory / f"{method}_status.csv").exists():
                    selected_files.add(directory / f"{method}_status.csv")
                states, records_by_rep = {}, {}
                if is_chat:
                    config = metadata["generation"]
                    mode = config["reasoning_effort"] or config["thinking"]
                    model = metadata["model"]
                    series = f"{model}/{method}/{mode}/{config['decoding_profile']}/p{metadata['protocol_hash']}"
                    request_path = directory / "requests" / f"{method}.json"
                    if series in requested:
                        raise ValueError("Duplicate protocol/method input; do not count copied trajectories as independent samples.")
                    saved_population = validate_saved_method_for_analysis(directory, metadata, method, feedback)
                    requested[series] = min(args.replicates, saved_population)
                    selected_files.add(request_path)
                    for checkpoint in sorted((directory / "checkpoints" / method).glob("replicate_*.json")):
                        state = llm_replay_io.read_json(checkpoint)
                        replicate = state["replicate"]
                        if replicate >= args.replicates:
                            continue
                        llm_replay_io.validate_checkpoint(state, metadata, method, replicate, feedback)
                        states[replicate] = state
                        selected_files.add(checkpoint)
                        logs = sorted((directory / "llm_logs" / method / f"replicate_{replicate:03d}").glob("*.json"))
                        records_by_rep[replicate] = [json.loads(p.read_text()) for p in logs]
                        for record in records_by_rep[replicate]:
                            if record["protocol_hash"] != metadata["protocol_hash"] or record["method"] != method or record["replicate"] != replicate:
                                raise ValueError("Offline request journal identity mismatch.")
                            if "response" in record and llm_client.digest(record["response"]) != record["response_sha256"]:
                                raise ValueError("Offline request journal checksum mismatch.")
                        selected_files.update(logs)
                    # Checkpoints are authoritative if an interruption preceded the CSV export.
                    table = pd.DataFrame([dict(method=method, replicate=r, seed=s["seed"], trial=i,
                                                experiment=row["experiment"], val_accuracy=float(row["val_accuracy"]),
                                                val_f1=float(row["val_f1"]), landscape="full")
                                          for r, s in states.items() for i, row in enumerate(s["rows"], 1)],
                                         columns=["method", "replicate", "seed", "trial", "experiment", "val_accuracy", "val_f1", "landscape"])
                else:
                    model, series = ((metadata["model"], f"{metadata['model']}/{method}/historical/s{metadata['snapshot_sha256']}")
                                     if is_historical else ("classical", f"{method}/p{metadata['protocol_hash']}"))
                    if series in requested:
                        raise ValueError("Duplicate protocol/method input; do not count copied trajectories as independent samples.")
                    table = baseline_io.validate_trials(
                        pd.read_csv(StringIO(archive[path].decode("utf-8"))) if is_historical else pd.read_csv(path),
                        method, args.budget, "full", None if is_historical else metadata["base_seed"], feedback)
                    saved_population = int(table["replicate"].nunique())
                    table = table.loc[table["replicate"].lt(args.replicates)].copy()
                    states = {int(r): dict(status="completed") for r in table["replicate"].unique()}
                    requested[series] = len(states)
                if is_chat:
                    for replicate in range(requested[series]):
                        states.setdefault(replicate, dict(status="not_started"))
                input_kind = "historical_csv" if is_historical else "local_chat" if is_chat else "classical_csv"
                population_scope = "saved completed trajectories only; original failure denominator unknown" if is_historical else "recorded trajectories"
                population_source = "request_ledger" if is_chat else "historical_completed_csv" if is_historical else "completed_csv"
                coverage.append(dict(series=series, model=model, method=method, protocol_hash=metadata["protocol_hash"],
                                     input_kind=input_kind, snapshot_sha256=metadata.get("snapshot_sha256"),
                                     input_directory=str(directory), population_source=population_source,
                                     saved_population=saved_population, included_replicates=requested[series],
                                     excluded_by_limit=saved_population - requested[series], replicate_id_limit=args.replicates))
                population_label = "ledger-requested trajectories" if is_chat else "completed CSV trajectories"
                print(f"Saved {model}/{method}: including {requested[series]} of {saved_population} {population_label}.")
                if requested[series] < saved_population:
                    print(f"  --replicates {args.replicates} excludes saved replicate IDs >= {args.replicates} "
                          "from this report. Raise this ceiling to include them; no trajectories are generated.")
                if not states:
                    continue
                conditions.append(dict(series=series, model=model, method=method, protocol_hash=metadata["protocol_hash"],
                                       input_kind=input_kind, snapshot_sha256=metadata.get("snapshot_sha256"),
                                       input_directory=str(directory), condition="historical; unrecorded" if is_historical else metadata.get("condition") or "default",
                                       thinking=config["thinking"] if is_chat else "unrecorded" if is_historical else "n/a",
                                       reasoning_effort=(config["reasoning_effort"] or "n/a") if is_chat else "unrecorded" if is_historical else "n/a",
                                       decoding_profile=config["decoding_profile"] if is_chat else "unrecorded" if is_historical else "n/a"))
                table = table.assign(model=model, series=series)
                if not table.empty:
                    all_frames.append(table)
                complete_ids = [r for r, s in states.items() if s["status"] == "completed"]
                completed = table.loc[table["replicate"].isin(complete_ids)].copy()
                if not completed.empty:
                    frames.append(completed)
                sequences, first = [], []
                for replicate, state in sorted(states.items()):
                    sub = table.loc[table["replicate"].eq(replicate)].sort_values("trial")
                    names = sub["experiment"].tolist()
                    provenance = metadata.get("server", {}).get("provenance", {})
                    gpu_count = provenance.get("inference_gpu_count", {}).get("value")
                    cost = request_costs(records_by_rep.get(replicate, []), gpu_count)
                    if is_historical:
                        cost = {key: np.nan for key in cost}
                    incumbent = sub.iloc[int(np.argmax(sub["val_accuracy"].to_numpy()))]["experiment"] if names else None
                    row = dict(series=series, model=model, method=method, replicate=replicate, status=state["status"],
                               population_scope=population_scope, failure_accounting="unavailable" if is_historical else "recorded",
                               api_endpoint=metadata.get("server", {}).get("endpoint", "unavailable" if is_historical else "not applicable"),
                               revision_provenance=provenance.get("revision", {}).get("status", "unavailable" if is_chat or is_historical else "not applicable"),
                               sampling_provenance=provenance.get("effective_sampling", {}).get("status", "unavailable" if is_chat or is_historical else "not applicable"),
                               reasoning_control_provenance=provenance.get("reasoning_control_rendered", {}).get("status", "unavailable" if is_chat or is_historical else "not applicable"),
                               n_evaluations=len(names), found_best=bool(set(names) & best), found_top3=bool(set(names) & top3),
                               best_val_accuracy=float(full_index.loc[names, "val_accuracy"].max()) if names else np.nan,
                               simple_regret=float(optimum - full_index.loc[names, "val_accuracy"].max()) if names else np.nan,
                               training_gpu_hours=float(full_index.loc[names, "runtime_hours"].sum()) if names else 0.0,
                               incumbent_experiment=incumbent,
                               incumbent_test_accuracy=float(full_index.loc[incumbent, "test_accuracy"]) if incumbent else np.nan,
                               incumbent_test_f1=float(full_index.loc[incumbent, "test_f1"]) if incumbent else np.nan, **cost)
                    terminal.append(row)
                    if names:
                        sequences.append(tuple(names))
                        first.append(names[0])
                frequencies = pd.Series(first, dtype=str).value_counts(normalize=True)
                diversity.append(dict(series=series, n_started=sum(s["status"] != "not_started" for s in states.values()), n_nonempty=len(sequences),
                                      n_unique_ordered_trajectories=len(set(sequences)),
                                      n_unique_selected_sets=len({frozenset(s) for s in sequences}),
                                      duplicate_ordered_prefix_rate=1 - len(set(sequences)) / len(sequences) if sequences else np.nan,
                                      duplicate_selected_set_rate=1 - len({frozenset(s) for s in sequences}) / len(sequences) if sequences else np.nan,
                                      n_completed=len(complete_ids),
                                      completed_duplicate_trajectory_rate=(1 - completed.groupby("replicate")["experiment"].agg(tuple).nunique() / len(complete_ids)) if complete_ids else np.nan,
                                      n_unique_first=len(frequencies),
                                      first_choice_entropy_bits=float(-(frequencies * np.log2(frequencies)).sum()),
                                      first_choice_frequencies=json.dumps(frequencies.to_dict(), sort_keys=True)))
            sources.append(dict(directory=str(directory), selected_methods=list(input_methods), metadata=metadata,
                                files={str(p.relative_to(directory)): hashlib.sha256(archive[p] if is_historical else p.read_bytes()).hexdigest()
                                       for p in sorted(selected_files)}))
    missing = set(args.methods) - {row["method"] for row in coverage}
    if missing:
        raise ValueError(f"No saved input found for requested methods: {', '.join(sorted(missing))}. "
                         "Select their existing protocol directories with --analysis-inputs; offline analysis never generates results.")
    if not terminal:
        raise ValueError("No matching trajectories in the explicitly selected protocols.")
    manifest = dict(protocol="llm-replay-offline-comparison", primary_metric="p_found_best_at_budget",
                    supporting_metric="mean_best_val_accuracy", principal_comparator="rf_ei",
                    methods=args.methods, replicate_limit=args.replicates, budget=args.budget, sources=sources,
                    input_coverage=coverage,
                    historical_population="explicit historical archives are analysed as saved completed trajectories only; original failure denominators and execution protocols are unknown; snapshot hashes are not protocol fingerprints",
                    curves="all_trajectory curves retain failures in denominators and carry stopped incumbents forward; completed_trajectory_curves are conditional supplementary plots only",
                    primary_intervals="95% Wilson discovery intervals; 2000 percentile bootstrap resamples of independent trajectory indices, seed 0; final estimates withheld until all requested IDs are terminal",
                    training_cost="fixed landscape training runtime on one L40S, in GPU-hours; separate from inference hardware/time",
                    monetary_cost="unavailable; no price or billing telemetry supplied",
                    inference_cost="request wall time / 3600; GPU-hours estimated only if server GPU parallelism is reported; excludes startup, rendering and idle time; not money or utilisation",
                    analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    reporting_source_hashes=reporting_source_hashes(Path(__file__).parent),
                    baseline_analysis_sha256=hashlib.sha256(Path(baseline_analysis.__file__).read_bytes()).hexdigest())
    output = runner.BASE_OUTPUT_DIR / "llm_comparisons" / f"inputs_{llm_client.digest(manifest)[:12]}"
    output.mkdir(parents=True, exist_ok=True)
    llm_replay_io.atomic_json(output / "comparison_metadata.json", manifest)
    terminal = pd.DataFrame(terminal)
    summary = primary_summary(terminal, requested, optimum)
    curves, compute = all_trajectory_curves(all_frames, terminal, full, args.budget)
    diversity = pd.DataFrame(diversity)
    final, diagnostics = report_tables(summary, diversity, conditions, args.budget)
    for name, table in (("input_coverage.csv", pd.DataFrame(coverage)),
                        ("trajectory_endpoints.csv", terminal), ("primary_outcomes.csv", summary),
                        ("trajectory_diversity.csv", diversity), ("final_summary.csv", final),
                        ("inference_diagnostics.csv", diagnostics), ("all_trajectory_curves.csv", curves), ("all_trajectory_compute.csv", compute)):
        atomic_write_csv(output / name, list(table.columns), table.to_dict("records"))
    llm_replay_io.sync_directory(output)
    if frames:
        # Keep the classical entry point unchanged, but avoid its wide conditional
        # table competing with the unconditional endpoint report printed here.
        with redirect_stdout(StringIO()):
            baseline_analysis.analyze(frames, full, output / "completed_trajectory_curves", manifest)
        print(f"Saved supplementary completed-only plots: {output / 'completed_trajectory_curves'}")
    print_report(final, diagnostics)
    print(f"Saved offline LLM replay comparison: {output}")
    return output
