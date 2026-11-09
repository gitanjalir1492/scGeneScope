"""Smoke-gated, append-only-in-meaning LLM trajectories for the local Chat path.

Checkpoints are authoritative; CSVs are atomic exports of every saved prefix.
Historical files are never imported. Network attempts are journalled individually
so received answers survive interruption before their observation is committed.
"""

import csv
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote

import classical_search
import llm_client
import llm_search
import summarize_experiment
from results_io import atomic_write_csv
from search_space import build_experiment_name


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path):
    if not path.exists():
        durable_mkdir(path.parent)
        path.mkdir(exist_ok=True)
        sync_directory(path.parent)


def atomic_json(path, value):
    durable_mkdir(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def readable_path(args, config):
    number = lambda value: str(value).removesuffix(".0")
    mode = f"thinking_{config['thinking']}"
    if config["reasoning_effort"]:
        mode += f"_effort_{config['reasoning_effort']}"
    sampling = "_".join(f"{label}{number(config[key])}" for label, key in (
        ("t", "temperature"), ("p", "top_p"), ("k", "top_k"), ("pp", "presence_penalty"),
        ("fp", "frequency_penalty"), ("rp", "repetition_penalty"), ("minp", "min_p"), ("out", "max_tokens")))
    relative = (Path(quote(config["model"], safe="-_.")) / args.landscape / f"budget_{args.budget:03d}"
                / f"seed_{args.seed}" / mode / f"{config['decoding_profile']}_{sampling}")
    return relative / f"condition_{args.condition}" if args.condition else relative


def protocol(args, config, server, search_dir, landscape_path, variants, functions):
    metadata = dict(schema_version=3, protocol="local-chat-replay", api="chat/completions",
                    model=config["model"], landscape=args.landscape, budget=args.budget, base_seed=args.seed,
                    condition=args.condition,
                    feedback_decimals=6, generation=config, server=server, variants=variants,
                    python=platform.python_version(),
                    landscape_sha256=hashlib.sha256(landscape_path.read_bytes()).hexdigest(),
                    primary_metric="optimum_discovery_at_budget", supporting_metric="mean_best_val_accuracy",
                    principal_comparator="rf_ei", other_comparators=["tpe_discrete", "gp_ei", "multimodal_diverse"],
                    seed_scheme="sha256(JSON [llm-replay-request-v1, base_seed, method[:summary], replicate, trial, attempt])[:8]; uint32; collisions refused before generation",
                    replicate_seed_scheme="base_seed + zero_based_replicate; catalogue shuffling only",
                    sampling_interpretation="distinct request seeds for stochastic trajectories; journal recovery reuses exact responses; server determinism and independence are not verified",
                    historical_reuse=False, source_hashes={})
    for name in ("llm_client.py", "llm_replay_io.py", "llm_search.py", "summarize_experiment.py", "search_space.py", "results_io.py", "replay_search.py", "classical_search.py"):
        metadata["source_hashes"][name] = hashlib.sha256((search_dir / name).read_bytes()).hexdigest()
    for function in functions:
        metadata["source_hashes"][function.__name__] = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    metadata["prompt_source_hashes"] = {name: metadata["source_hashes"][name] for name in ("llm_search.py", "summarize_experiment.py", "search_space.py")}
    metadata["initial_prompt_sha256"] = {
        method: llm_client.digest(llm_search.build_prompt(
            [], variant["mode"], shuffle_seed=args.seed if variant["shuffle"] else None,
            budget=args.budget if variant["budget"] else None,
            isolation_principles=variant["isolation"], gpu_hours_aware=variant["gpu"])[0])
        for method, variant in variants.items()}
    metadata["prompt_hash_definition"] = "SHA256 of compact JSON-encoded prompt string; every actual prompt hash is also saved in its request journal"
    metadata["reproducibility_limitations"] = [
        "API acceptance does not verify internal seed, sampling or reasoning-control application.",
        "Unreported checkpoint revisions and server settings cannot be compared across restarts.",
        "Journal recovery is deterministic reuse; distinct seeded requests do not guarantee independent or bitwise-reproducible server sampling.",
        *server.get("warnings", [])]
    metadata["protocol_hash"] = llm_client.digest(metadata)
    return readable_path(args, config), metadata


def read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable or corrupt saved JSON: {path}; preserving it for inspection.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a saved JSON object: {path}; preserving it for inspection.")
    return value


def response_identity(raw):
    fingerprint = raw.get("system_fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise ValueError("Malformed response system_fingerprint.")
    return dict(system_fingerprint=llm_client.evidence(
        fingerprint, "Chat response system_fingerprint; server report, not verification of weights"))


def metadata_differences(saved, requested, prefix=""):
    differences = []
    for key in sorted(set(saved) | set(requested)):
        if key == "protocol_hash":
            continue
        name = f"{prefix}.{key}" if prefix else key
        old, new = saved.get(key), requested.get(key)
        if key not in saved or key not in requested:
            differences.append(f"{name}: field added or removed")
        elif isinstance(old, dict) and isinstance(new, dict):
            differences.extend(metadata_differences(old, new, name))
        elif old != new:
            differences.append(f"{name}: saved={old!r}, requested={new!r}")
    return differences


@contextmanager
def writer(directory, metadata):
    """Lock the readable condition and fail closed on incompatible/orphaned data."""
    durable_mkdir(directory)
    with (directory / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another replay is writing {directory}") from error
        path = directory / "metadata.json"
        if path.exists():
            saved = read_json(path)
            if llm_client.digest({k: v for k, v in saved.items() if k != "protocol_hash"}) != saved.get("protocol_hash"):
                raise ValueError(f"Invalid metadata fingerprint in {directory}; refusing to overwrite.")
            if saved != metadata:
                differences = "\n  ".join(metadata_differences(saved, metadata))
                raise ValueError(f"Incompatible protocol in {directory}; refusing to resume or overwrite.\n  {differences}\n"
                                 "Use a separately named --condition and validate a new smoke; no hash will be appended.")
        else:
            if any(p.name != ".writer.lock" for p in directory.iterdir()):
                raise ValueError(f"Existing files in {directory} have no metadata; refusing to adopt them.")
            atomic_json(path, metadata)
        yield


def save_checkpoint(path, state):
    state["state_sha256"] = llm_client.digest({k: v for k, v in state.items() if k != "state_sha256"})
    atomic_json(path, state)


def validate_checkpoint(state, metadata, method, replicate, feedback=None):
    if type(replicate) is not int or replicate < 0:
        raise ValueError("Invalid checkpoint replicate index.")
    if state.get("state_sha256") != llm_client.digest({k: v for k, v in state.items() if k != "state_sha256"}):
        raise ValueError("Checkpoint checksum mismatch; preserving it for inspection.")
    if (state["protocol_hash"], state["method"], state["replicate"], state["seed"]) != (
            metadata["protocol_hash"], method, replicate, metadata["base_seed"] + replicate):
        raise ValueError("Checkpoint identity mismatch; preserving it for inspection.")
    rows = state["rows"]
    if not 0 <= len(rows) <= metadata["budget"] or state["status"] not in {"in_progress", "completed", "failed"}:
        raise ValueError("Invalid checkpoint status or budget.")
    if state["status"] == "completed" and len(rows) != metadata["budget"]:
        raise ValueError("Completed checkpoint does not have the full budget.")
    used = set()
    for number, row in enumerate(rows, 1):
        config = classical_search.configuration(row)
        experiment = build_experiment_name(config)
        if row["experiment"] != experiment or experiment in used or row["status"] != "completed":
            raise ValueError("Invalid or duplicate checkpoint observation.")
        if row["experiment_id"] != f"{method}_{number:03d}":
            raise ValueError("Checkpoint observation order mismatch.")
        if row["search_method"] != llm_search.method_name(metadata["variants"][method]["mode"]):
            raise ValueError("Checkpoint prompt history method mismatch.")
        used.add(experiment)
        for metric, decimals in (("val_accuracy", 6), ("val_f1", 6), ("runtime_hours", 4)):
            value = float(row[metric])
            if not 0 <= value < float("inf") or (metric != "runtime_hours" and value > 1):
                raise ValueError("Invalid checkpoint feedback.")
            if feedback is not None and row[metric] != f"{feedback[experiment][metric]:.{decimals}f}":
                raise ValueError("Checkpoint feedback differs from selected evaluator results.")
    return state


def export_method(directory, metadata, method):
    rows, statuses = [], []
    for path in sorted((directory / "checkpoints" / method).glob("replicate_*.json")):
        state = read_json(path)
        validate_checkpoint(state, metadata, method, state["replicate"])
        statuses.append(dict(method=method, replicate=state["replicate"], seed=state["seed"],
                             status=state["status"], n_evaluations=len(state["rows"]), error=state.get("error", "")))
        for trial, row in enumerate(state["rows"], 1):
            rows.append(dict(method=method, replicate=state["replicate"], seed=state["seed"], trial=trial,
                             experiment=row["experiment"], val_accuracy=float(row["val_accuracy"]),
                             val_f1=float(row["val_f1"]), selection_reason=row["selection_reason"],
                             experiment_summary=row["experiment_summary"], landscape=metadata["landscape"]))
    fields = ["method", "replicate", "seed", "trial", "experiment", "val_accuracy", "val_f1",
              "selection_reason", "experiment_summary", "landscape"]
    atomic_write_csv(directory / f"{method}.csv", fields, rows)
    atomic_write_csv(directory / f"{method}_status.csv",
                     ["method", "replicate", "seed", "status", "n_evaluations", "error"], statuses)
    sync_directory(directory)


def check_existing_export(directory, metadata, method, feedback):
    path = directory / f"{method}.csv"
    if not path.exists():
        return
    seen = set()
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            replicate, trial = int(row["replicate"]), int(row["trial"])
            checkpoint = directory / "checkpoints" / method / f"replicate_{replicate:03d}.json"
            if not checkpoint.exists() or (replicate, trial) in seen:
                raise ValueError("Saved CSV lacks its checkpoint or contains duplicates; refusing to overwrite.")
            seen.add((replicate, trial))
            state = validate_checkpoint(read_json(checkpoint), metadata, method, replicate, feedback)
            if not 1 <= trial <= len(state["rows"]):
                raise ValueError("CSV has observations absent from the checkpoint.")
            expected = state["rows"][trial - 1]
            if row["method"] != method or int(row["seed"]) != state["seed"] or row["landscape"] != metadata["landscape"]:
                raise ValueError("CSV identity mismatch.")
            for field in ("experiment", "selection_reason", "experiment_summary"):
                # A pending summary may have been committed after the last CSV export.
                if field == "experiment_summary" and not row[field]:
                    continue
                if row[field] != expected[field]:
                    raise ValueError("CSV differs from its authoritative checkpoint.")
            if any(float(row[k]) != float(expected[k]) for k in ("val_accuracy", "val_f1")):
                raise ValueError("CSV feedback differs from its checkpoint.")


def validate_journal(path, record, metadata, method, replicate):
    if (record.get("protocol_hash"), record.get("method"), record.get("replicate")) != (metadata["protocol_hash"], method, replicate):
        raise ValueError(f"Journal identity mismatch: {path}")
    trial, attempt, kind = record.get("trial"), record.get("attempt"), record.get("kind")
    config = metadata["generation"]
    if (type(trial) is not int or not 1 <= trial <= metadata["budget"] or type(attempt) is not int
            or not 1 <= attempt <= config["selection_attempts"] or kind not in {"decision", "summary"}
            or (kind == "summary" and method != "llm_summary")):
        raise ValueError(f"Invalid journal coordinates: {path}")
    stem = f"trial_{trial:03d}_{kind}_attempt_{attempt:02d}_transport_"
    suffix = path.stem.removeprefix(stem)
    if not path.stem.startswith(stem) or not suffix.isdigit() or f"{int(suffix):04d}" != suffix:
        raise ValueError(f"Journal filename/identity mismatch: {path}")
    seed = llm_client.request_seed(metadata["base_seed"], method if kind == "decision" else method + ":summary", replicate, trial, attempt)
    body = record.get("request", {})
    messages = body.get("messages", [])
    if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict) or not isinstance(messages[0].get("content"), str):
        raise ValueError(f"Malformed journal request: {path}")
    prompt = messages[0]["content"]
    if (body != llm_client.request_body(prompt, config, seed) or record.get("prompt_sha256") != llm_client.digest(prompt)
            or record.get("request_seed") != seed or record.get("replicate_seed") != metadata["base_seed"] + replicate):
        raise ValueError(f"Journal request/settings/seed mismatch: {path}")
    status = record.get("status")
    if status not in {"pending", "transport_error", "received", "accepted", "invalid_output"}:
        raise ValueError(f"Invalid journal status: {path}")
    if status in {"received", "accepted", "invalid_output"} and "response" not in record:
        raise ValueError(f"Missing journal response: {path}")
    if "response" in record and record.get("response_sha256") != llm_client.digest(record["response"]):
        raise ValueError(f"Journal response checksum mismatch: {path}")


def validate_saved_method(directory, metadata, method, feedback=None):
    """Validate all saved IDs, including IDs outside a reduced requested count."""
    states = {}
    for path in sorted((directory / "checkpoints" / method).glob("*.json")):
        state = read_json(path)
        replicate = state.get("replicate")
        validate_checkpoint(state, metadata, method, replicate, feedback)
        if path.name != f"replicate_{replicate:03d}.json" or replicate in states:
            raise ValueError(f"Checkpoint filename/identity mismatch: {path}")
        states[replicate] = state
    for log_directory in (directory / "llm_logs" / method).glob("*"):
        if not log_directory.is_dir() or not any(log_directory.name == f"replicate_{r:03d}" for r in states):
            raise ValueError(f"Orphaned inference logs without a checkpoint: {log_directory}")
    for replicate, state in states.items():
        accepted = {}
        for path in sorted((directory / "llm_logs" / method / f"replicate_{replicate:03d}").glob("*.json")):
            record = read_json(path)
            validate_journal(path, record, metadata, method, replicate)
            trial, kind = record["trial"], record["kind"]
            if trial > len(state["rows"]) + (kind == "decision"):
                raise ValueError(f"Journal is ahead of its saved prefix: {path}")
            if record["status"] == "accepted":
                identity = read_json(directory / "response_identity.json")
                if identity != response_identity(record["response"]):
                    raise ValueError("Accepted response fingerprint differs from the saved server response identity.")
                key = (trial, kind)
                if key in accepted:
                    raise ValueError(f"Multiple accepted responses for one selection: {path}")
                accepted[key] = record
        for trial, row in enumerate(state["rows"], 1):
            record = accepted.get((trial, "decision"))
            if record is None:
                raise ValueError("Checkpoint observation lacks its accepted decision journal.")
            variant = metadata["variants"][method]
            prompt, configs, available = llm_search.build_prompt(
                state["rows"][:trial - 1], variant["mode"],
                shuffle_seed=state["seed"] if variant["shuffle"] else None,
                budget=metadata["budget"] if variant["budget"] else None,
                isolation_principles=variant["isolation"], gpu_hours_aware=variant["gpu"])
            config_id, reason, _, _ = llm_search.parse_response(llm_client.normalize_response(record["response"])["text"], variant["mode"])
            if (record["request"]["messages"][0]["content"] != prompt or config_id not in available
                    or build_experiment_name(configs[config_id]) != row["experiment"] or reason != row["selection_reason"]):
                raise ValueError("Checkpoint decision differs from its observed-history request/answer.")
            if row["experiment_summary"]:
                summary = accepted.get((trial, "summary"))
                if summary is None or summarize_experiment.parse_summary(llm_client.normalize_response(summary["response"])["text"]) != row["experiment_summary"]:
                    raise ValueError("Checkpoint summary differs from its accepted journal.")
                if summary["request"]["messages"][0]["content"] != summarize_experiment.build_prompt(state["rows"][:trial], row):
                    raise ValueError("Checkpoint summary request differs from its observed history.")
            elif method == "llm_summary" and (trial < len(state["rows"]) or state["status"] == "completed"):
                raise ValueError("Missing summary in a completed checkpoint prefix.")
    check_existing_export(directory, metadata, method, feedback)
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
                    raise ValueError("Saved status CSV differs from its checkpoints; refusing to overwrite.")
                seen.add(replicate)
    requested = directory / "requests" / f"{method}.json"
    count = read_json(requested).get("replicates") if requested.exists() else 0
    if type(count) is not int or count < 0 or (states and count <= max(states)):
        raise ValueError("Missing or invalid requested-replicate ledger; refusing to lose the failure denominator.")
    return count


def validate_seed_schedule(metadata, counts):
    seen = {}
    for method, count in sorted(counts.items()):
        kinds = [method, method + ":summary"] if method == "llm_summary" else [method]
        for seed_method in kinds:
            for replicate in range(count):
                for trial in range(1, metadata["budget"] + 1):
                    for attempt in range(1, metadata["generation"]["selection_attempts"] + 1):
                        identity = (seed_method, replicate, trial, attempt)
                        seed = llm_client.request_seed(metadata["base_seed"], *identity)
                        if seed in seen:
                            raise ValueError(f"Request seed collision: {seen[seed]} and {identity} use {seed}. Choose a different base seed and validate a new smoke; saved results are unchanged.")
                        seen[seed] = identity


class SelectionFailure(RuntimeError):
    """All predeclared model-output attempts failed; do not replace this replicate."""


class Journal:
    def __init__(self, directory, metadata, method, replicate, client):
        self.protocol_directory = directory
        self.directory = directory / "llm_logs" / method / f"replicate_{replicate:03d}"
        self.metadata, self.method, self.replicate, self.client = metadata, method, replicate, client

    def response(self, body, trial, attempt, kind, rendered):
        stem = f"trial_{trial:03d}_{kind}_attempt_{attempt:02d}_transport_"
        existing = sorted(self.directory.glob(stem + "*.json"))
        for path in existing:
            record = read_json(path)
            validate_journal(path, record, self.metadata, self.method, self.replicate)
            if record["request"] != body or record["protocol_hash"] != self.metadata["protocol_hash"]:
                raise ValueError("Journal request/protocol mismatch.")
            if "response" in record:
                if record["response_sha256"] != llm_client.digest(record["response"]):
                    raise ValueError("Journal response checksum mismatch.")
                return path, record
        # Interrupted in-flight calls remain visible; a manual resume repeats their
        # seed/payload. Transport retries do not secretly resample the policy.
        for retry in range(self.metadata["generation"]["transport_attempts_per_invocation"]):
            path = self.directory / f"{stem}{len(existing) + retry:04d}.json"
            record = dict(protocol_hash=self.metadata["protocol_hash"], method=self.method, replicate=self.replicate,
                          trial=trial, attempt=attempt, kind=kind, request=body, status="pending",
                          replicate_seed=self.metadata["base_seed"] + self.replicate, request_seed=body["seed"],
                          prompt_sha256=llm_client.digest(body["messages"][0]["content"]),
                          prompt_tokens_preflight=rendered["count"], prompt_token_sha256=rendered["token_sha256"],
                          effective_sampling=rendered["sampling"],
                          started_at=time.time())
            atomic_json(path, record)
            started = time.perf_counter()
            try:
                raw = self.client.complete(body)
            except Exception as error:
                transient = isinstance(error, (URLError, TimeoutError, ConnectionError))
                if isinstance(error, HTTPError):
                    transient = error.code in {408, 429, 500, 502, 503, 504}
                record.update(status="transport_error", error_type=type(error).__name__, error=str(error),
                              retryable=transient, latency_seconds=time.perf_counter() - started)
                if isinstance(error, HTTPError):
                    record["http_status"] = error.code
                    record["error_body"] = error.read(8192).decode("utf-8", errors="replace") if error.fp is not None else ""
                atomic_json(path, record)
                if not transient or retry + 1 == self.metadata["generation"]["transport_attempts_per_invocation"]:
                    raise RuntimeError("Inference interrupted; prefix and request journal preserved. Inspect the logged API error; no parameters were dropped or changed.") from error
                time.sleep(2 ** retry)
                continue
            record.update(status="received", response=raw, response_sha256=llm_client.digest(raw),
                          latency_seconds=time.perf_counter() - started)
            atomic_json(path, record)
            return path, record
        raise AssertionError("Unreachable retry state")

    def ask(self, prompt, trial, parse, kind="decision"):
        config = self.metadata["generation"]
        context = self.metadata["server"]["model"]["max_model_len"]
        for attempt in range(1, config["selection_attempts"] + 1):
            seed_method = self.method if kind == "decision" else self.method + ":summary"
            seed = llm_client.request_seed(self.metadata["base_seed"], seed_method, self.replicate, trial, attempt)
            rendered = self.client.render(prompt, seed=seed)
            if context is not None and rendered["count"] is not None and rendered["count"] + config["max_tokens"] + llm_client.HEADROOM > context:
                raise ValueError("Prompt plus fixed generation cap exceeds context headroom. No truncation or cap adjustment is allowed.")
            # An optional renderer may disappear, but previously verified
            # evidence cannot silently become an unverified production claim.
            required_render = self.metadata["server"].get("provenance", {}).get("effective_sampling", {}).get("status") == "verified"
            if required_render and not llm_client.check_sampling(rendered["sampling"], config, seed):
                raise ValueError("Previously verified render evidence is now unavailable; preserving the trajectory for inspection.")
            body = llm_client.request_body(prompt, config, seed)
            path, record = self.response(body, trial, attempt, kind, rendered)
            try:
                normalized = llm_client.normalize_response(record["response"])
                returned_model = record["response"].get("model")
                if returned_model is not None and returned_model not in {config["model"], config["model_id"]}:
                    raise ValueError("Chat response model differs from the requested model.")
                # Some standard Chat servers expose this even without diagnostics.
                # Freeze the reported value; it still does not verify model weights.
                identity_path = self.protocol_directory / "response_identity.json"
                identity = response_identity(record["response"])
                if identity_path.exists():
                    if read_json(identity_path) != identity:
                        raise RuntimeError("Chat response system_fingerprint changed; preserving the journal and stopping this protocol.")
                else:
                    atomic_json(identity_path, identity)
                usage = normalized["usage"]
                if type(usage.get("completion_tokens")) is int and usage["completion_tokens"] > config["max_tokens"]:
                    raise ValueError("Response exceeds the fixed requested output-token limit.")
                if context is not None and type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] + config["max_tokens"] + llm_client.HEADROOM > context:
                    raise ValueError("Actual prompt usage has insufficient headroom for the fixed output cap.")
                parsed = parse(normalized["text"])
                if config["thinking"] == "off" and normalized["reasoning"].strip():
                    raise ValueError("Reasoning returned despite thinking-off; stop and audit server controls.")
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                record.update(status="invalid_output", validation_error=str(error))
                atomic_json(path, record)
                continue
            record.update(status="accepted", normalized=normalized)
            atomic_json(path, record)
            return parsed
        raise SelectionFailure(f"No valid {kind} after {config['selection_attempts']} attempts.")


def run_replicate(directory, metadata, method, replicate, feedback, client, base_row, complete):
    path = directory / "checkpoints" / method / f"replicate_{replicate:03d}.json"
    if path.exists():
        state = validate_checkpoint(read_json(path), metadata, method, replicate, feedback)
    else:
        state = dict(protocol_hash=metadata["protocol_hash"], method=method, replicate=replicate,
                     seed=metadata["base_seed"] + replicate, status="in_progress", rows=[])
        if (directory / "llm_logs" / method / f"replicate_{replicate:03d}").exists():
            raise ValueError("Orphaned inference logs without a checkpoint; refusing regeneration.")
        save_checkpoint(path, state)
    if state["status"] != "in_progress":
        label = "completed; no inference sent" if state["status"] == "completed" else "terminally failed; no replacement attempted"
        print(f"{method}: replicate {replicate + 1} reused ({label})")
        return state
    if state["rows"]:
        print(f"{method}: replicate {replicate + 1} resuming after {len(state['rows'])} accepted evaluations")
    journal = Journal(directory, metadata, method, replicate, client)
    rows = state["rows"]
    variant = metadata["variants"][method]
    try:
        while True:
            if method == "llm_summary" and rows and not rows[-1]["experiment_summary"]:
                prompt = summarize_experiment.build_prompt(rows, rows[-1])
                rows[-1]["experiment_summary"] = journal.ask(prompt, len(rows), summarize_experiment.parse_summary, "summary")
                save_checkpoint(path, state)
            if len(rows) == metadata["budget"]:
                break
            prompt, configs, available = llm_search.build_prompt(
                rows, variant["mode"], shuffle_seed=state["seed"] if variant["shuffle"] else None,
                budget=metadata["budget"] if variant["budget"] else None,
                isolation_principles=variant["isolation"], gpu_hours_aware=variant["gpu"])
            def parse(text):
                config_id, reason, _, _ = llm_search.parse_response(text, variant["mode"])
                if config_id not in available:
                    raise ValueError(f"Invalid or previously selected configuration ID: {config_id}")
                return configs[config_id], reason
            config, reason = journal.ask(prompt, len(rows) + 1, parse)
            row = base_row(config, method, len(rows) + 1, reason)
            complete(row, feedback)  # The evaluator sees only the already selected architecture.
            rows.append(row)
            save_checkpoint(path, state)
            validate_checkpoint(read_json(path), metadata, method, replicate, feedback)
            export_method(directory, metadata, method)
        state["status"] = "completed"
    except SelectionFailure as error:
        state.update(status="failed", error=str(error))
    except BaseException as error:
        state.setdefault("interruptions", []).append(dict(time=time.time(), error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        save_checkpoint(path, state)
        export_method(directory, metadata, method)
    return state


def smoke_evidence(directory, metadata, method):
    checkpoint = directory / "checkpoints" / method / "replicate_000.json"
    state = validate_checkpoint(read_json(checkpoint), metadata, method, 0)
    if state["status"] != "completed":
        raise ValueError("Smoke trajectory did not complete. Production remains blocked.")
    paths = sorted((directory / "llm_logs" / method / "replicate_000").glob("*.json"))
    records = [read_json(path) for path in paths]
    for path, record in zip(paths, records):
        validate_journal(path, record, metadata, method, 0)
    accepted = [r for r in records if r["status"] == "accepted"]
    expected = metadata["budget"] * (2 if method == "llm_summary" else 1)
    if len(accepted) != expected or any(r["status"] == "invalid_output" for r in records):
        raise ValueError("Smoke output validation failed; inspect the journal before changing the protocol.")
    config = metadata["generation"]
    reasoning = False
    sampling_verified = True
    rendering_verified = True
    context = metadata["server"]["model"]["max_model_len"]
    for record in accepted:
        normalized = llm_client.normalize_response(record["response"])
        usage = normalized["usage"]
        if not all(type(usage.get(k)) is int and usage[k] > 0 for k in ("prompt_tokens", "completion_tokens")):
            raise ValueError("Smoke requires measured input/output token usage.")
        if usage["completion_tokens"] >= config["max_tokens"]:
            raise ValueError("Smoke reached the output cap; production remains blocked.")
        if context is not None and usage["prompt_tokens"] + config["max_tokens"] + llm_client.HEADROOM > context:
            raise ValueError("Actual prompt token usage has insufficient headroom.")
        seed_method = method if record["kind"] == "decision" else method + ":summary"
        seed = llm_client.request_seed(metadata["base_seed"], seed_method, 0, record["trial"], record["attempt"])
        if record["request"]["seed"] != seed or record["request"] != llm_client.request_body(
                record["request"]["messages"][0]["content"], config, seed):
            raise ValueError("Smoke request settings or seeds differ from the declared protocol.")
        sampling_verified &= llm_client.check_sampling(record["effective_sampling"], config, seed)
        rendering_verified &= record["prompt_tokens_preflight"] is not None
        if record["prompt_tokens_preflight"] is not None and usage["prompt_tokens"] != record["prompt_tokens_preflight"]:
            raise ValueError("Actual prompt usage differs from the Chat rendering preflight.")
        reasoning |= bool(normalized["reasoning"].strip())
    if reasoning != (config["thinking"] == "on"):
        raise ValueError("Generated reasoning channels do not confirm the requested mode.")
    return dict(protocol_hash=metadata["protocol_hash"], method=method, replicate=0,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                response_identity=read_json(directory / "response_identity.json"),
                journal_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                checks=["chat_request_accepted", "generated_channels", "final_answer_schema", "valid_unique_ids",
                        "fixed_requested_output_cap", "explicit_request_seeds", "checkpoint_reload"],
                verification=dict(effective_sampling="verified" if sampling_verified else "unavailable",
                                  rendered_prompt_tokens="verified" if rendering_verified else "unavailable",
                                  context_headroom=metadata["server"].get("provenance", {}).get("context_window", {}).get("status", "verified" if context else "unavailable"),
                                  reasoning_control=metadata["server"].get("provenance", {}).get("reasoning_control_rendered", {}).get("status", "unavailable"),
                                  reasoning_channels="verified; observed in returned messages, not proof of internal effort"),
                limitations=metadata["server"].get("warnings", []),
                accepted_requests=len(accepted), transport_events=len(records) - len(accepted))


def run(args, runner):
    if not set(args.methods) <= set(runner.LLM_VARIANTS):
        raise ValueError("Chat generation accepts LLM methods only; run classical methods separately.")
    if args.landscape != "full":
        raise ValueError("Local LLM replay uses the completed full landscape.")
    if args.smoke_test and args.replicates != 1:
        raise ValueError("--smoke-test requires --replicates 1 per selected method.")
    config = llm_client.settings(args)
    client = llm_client.LocalClient(config)
    server = client.preflight()
    for warning in server.get("warnings", []):
        print(f"PROVENANCE: {warning}")
    relative, metadata = protocol(args, config, server, runner.SEARCH_DIR, runner.LANDSCAPES[args.landscape],
                                  runner.LLM_VARIANTS, (runner.base_row, runner.complete))
    directory = runner.BASE_REPLAY_DIR / "models" / relative
    print(f"LLM replay results: {directory}")
    with writer(directory, metadata):
        feedback = runner.load_landscape(args.landscape)
        saved_methods = set(args.methods)
        saved_methods.update(p.stem.removesuffix("_status") for p in directory.glob("*.csv"))
        for name in ("checkpoints", "llm_logs", "requests", "validation"):
            saved_methods.update(p.stem if p.is_file() else p.name for p in (directory / name).glob("*") if not p.name.endswith(".tmp"))
        if not saved_methods <= set(metadata["variants"]):
            raise ValueError("Unrecognized saved method; refusing to ignore existing files.")
        counts = {method: validate_saved_method(directory, metadata, method, feedback) for method in saved_methods}
        for method in saved_methods:
            gate = directory / "validation" / f"{method}.json"
            if gate.exists() and read_json(gate) != smoke_evidence(directory, metadata, method):
                raise ValueError(f"Production requires unchanged smoke evidence for {method}; refusing to revalidate altered files.")
            if method in args.methods and not args.smoke_test and not gate.exists():
                raise ValueError(f"Production requires an unchanged, validated smoke trajectory for {method}. Use --smoke-test --replicates 1.")
        counts.update({method: max(args.replicates, counts[method]) for method in args.methods})
        validate_seed_schedule(metadata, counts)
        for method in args.methods:
            requested = directory / "requests" / f"{method}.json"
            atomic_json(requested, dict(replicates=counts[method]))
        for method in args.methods:
            for replicate in range(args.replicates):
                state = run_replicate(directory, metadata, method, replicate, feedback, client, runner.base_row, runner.complete)
                print(f"{method}: saved status for replicate {replicate + 1}/{args.replicates}: {state['status']}")
            # Export even if all replicates were already complete, without filtering saved IDs.
            export_method(directory, metadata, method)
            if args.smoke_test:
                evidence = smoke_evidence(directory, metadata, method)
                gate = directory / "validation" / f"{method}.json"
                if not gate.exists():
                    atomic_json(gate, evidence)
                print(f"{method}: smoke validated; requested settings and recorded evidence frozen. Extend with the identical command without --smoke-test.")
                print(f"Verification: {json.dumps(evidence['verification'], sort_keys=True)}")
