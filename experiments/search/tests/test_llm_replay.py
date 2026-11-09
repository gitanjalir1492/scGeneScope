"""Synthetic local LLM replay checks: all inference/HTTP is mocked; no model loading."""

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import classical_search
import llm_analysis
import llm_client
import llm_replay_io as storage
import replay_search as replay
import replay_with_report
from search_space import build_experiment_name

HTTPClient = llm_client.LocalClient


def arguments(*extra):
    return replay.build_parser().parse_args([
        "--llm-api", "chat", "--llm-model", "qwen3-8b-thinking", "--methods", "llm_metrics",
        "--replicates", "1", "--seed", "2000", "--max-output-tokens", "1024", *extra])


def response(text, reasoning="synthetic reasoning", finish="stop"):
    return dict(id="synthetic", choices=[dict(message=dict(content=text, reasoning=reasoning), finish_reason=finish)],
                usage=dict(prompt_tokens=3000, completion_tokens=50, completion_tokens_details=dict(reasoning_tokens=30)))


class FakeClient:
    def __init__(self, config, calls, control):
        self.config, self.calls, self.control = config, calls, control

    def preflight(self):
        return dict(endpoint=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
                    vllm_version="synthetic", model=dict(max_model_len=32768),
                    control_probe=dict(requested="on", alternative="off"))

    def render(self, prompt, config=None, seed=0):
        return dict(count=3000, token_sha256=llm_client.digest(prompt), text=prompt,
                    sampling=dict(seed=seed, max_tokens=self.config["max_tokens"]))

    def complete(self, body):
        self.calls.append(body)
        fail_at = self.control.get("fail_at")
        if fail_at == len(self.calls):
            raise RuntimeError("synthetic server interruption")
        prompt = body["messages"][0]["content"]
        if prompt.startswith("You are maintaining scientific memory"):
            return response(json.dumps(dict(summary="Synthetic summary based on observed validation.")))
        available = prompt.rsplit("Available untested configurations:", 1)[-1]
        config_id = re.search(r"config_\d+", available).group()
        if self.control.get("invalid"):
            config_id = "config_nonexistent"
        return response(json.dumps(dict(configuration_id=config_id, reason="synthetic selection")),
                        "synthetic reasoning" if self.config["thinking"] == "on" else "")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    full = []
    for i, config in enumerate(classical_search.catalogue()):
        full.append(dict(experiment=build_experiment_name(config), status="completed", val_accuracy=0.2 + i / 100,
                         val_f1=0.15 + i / 100, runtime_hours=1 + i / 10, test_accuracy=0.8 - i / 100, test_f1=0.7 - i / 100))
    landscape = tmp_path / "synthetic_full.csv"
    with landscape.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(full[0]))
        writer.writeheader()
        writer.writerows(full)
    feedback = {r["experiment"]: {k: r[k] for k in ("val_accuracy", "val_f1", "runtime_hours")} for r in full}
    runner = SimpleNamespace(SEARCH_DIR=replay.SEARCH_DIR, LANDSCAPES={"full": landscape},
                             BASE_REPLAY_DIR=tmp_path / "results", BASE_OUTPUT_DIR=tmp_path / "analysis",
                             LLM_VARIANTS=replay.LLM_VARIANTS, base_row=replay.base_row, complete=replay.complete,
                             load_landscape=lambda name: feedback)
    calls, control = [], {}
    monkeypatch.setattr(llm_client, "LocalClient", lambda config: FakeClient(config, calls, control))
    return runner, calls, control


def directories(runner):
    return sorted(path.parent for path in (runner.BASE_REPLAY_DIR / "models").rglob("metadata.json"))


def test_original_responses_client_unchanged(monkeypatch):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text='{"configuration_id":"config_001","reason":"ok"}')
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: SimpleNamespace(responses=SimpleNamespace(create=create))))
    assert replay.llm_search.query_llm("unchanged prompt").startswith('{"configuration_id"')
    assert calls == [dict(model=replay.llm_search.LLM_MODEL, input="unchanged prompt",
                          max_output_tokens=replay.llm_search.LLM_MAX_OUTPUT_TOKENS)]
    defaults = replay.build_parser().parse_args([])
    assert defaults.llm_api == "responses" and defaults.replicates == 100
    assert defaults.methods == ["random", "bayesian"] and defaults.max_output_tokens is None


def test_qwen38_controlled_and_secondary_profiles():
    on = llm_client.settings(arguments("--llm-model", "qwen3.8-27b", "--thinking", "on"))
    off = llm_client.settings(arguments("--llm-model", "qwen3.8-27b", "--thinking", "off"))
    secondary = llm_client.settings(arguments("--llm-model", "qwen3.8-27b", "--thinking", "off",
                                            "--decoding-profile", "publisher-nonthinking"))
    fields = ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens")
    assert [on[k] for k in fields] == [off[k] for k in fields]
    assert [secondary[k] for k in fields[:4]] == [0.7, 0.8, 20, 1.5]
    assert on["reasoning_effort"] == "xhigh" and off["reasoning_effort"] is None
    assert len({llm_client.digest(c) for c in (on, off, secondary)}) == 3
    with pytest.raises(ValueError, match="override"):
        llm_client.settings(arguments("--llm-model", "qwen3.8-27b", "--thinking", "off",
                                      "--decoding-profile", "publisher-nonthinking", "--temperature", "1"))


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_gptoss_native_effort(effort):
    config = llm_client.settings(arguments("--llm-model", "gpt-oss-20b", "--reasoning-effort", effort))
    body = llm_client.request_body("original prompt", config, 7)
    assert body["reasoning_effort"] == effort and body["seed"] == 7
    assert body["messages"] == [dict(role="user", content="original prompt")]
    assert body["max_tokens"] == 1024
    assert body["chat_template_kwargs"] == {} and "response_format" not in body and "tools" not in body


@pytest.mark.parametrize("extra", [
    ["--llm-model", "gpt-oss-20b", "--thinking", "off"],
    ["--llm-model", "qwen3.8-27b", "--reasoning-effort", "high"],
    ["--llm-model", "qwen3.8-27b", "--thinking", "off", "--reasoning-effort", "low"],
    ["--temperature", "nan"], ["--top-p", "0"], ["--top-k", "0"], ["--presence-penalty", "3"],
    ["--model-revision", "main"], ["--max-output-tokens", "0"],
])
def test_invalid_settings_rejected(extra):
    with pytest.raises(ValueError):
        llm_client.settings(arguments(*extra))


def test_seed_schedule_and_reasoning_switch():
    seeds = {llm_client.request_seed(2000, "llm_metrics", r, t, a) for r in range(3) for t in range(1, 11) for a in range(1, 4)}
    assert len(seeds) == 90
    assert llm_client.request_seed(2000, "llm_metrics", 1, 2, 1) == llm_client.request_seed(2000, "llm_metrics", 1, 2, 1)
    off = llm_client.settings(arguments("--llm-model", "qwen3.8-27b", "--thinking", "off"))
    body = llm_client.request_body("prompt", off, 123)
    assert body["chat_template_kwargs"] == dict(enable_thinking=False, preserve_thinking=False)
    assert "reasoning_effort" not in body


def test_reasoning_and_final_are_separate():
    raw = response('{"configuration_id":"config_001","reason":"final"}', reasoning='{"configuration_id":"forbidden"}')
    assert llm_client.normalize_response(raw)["text"].endswith('"final"}')
    message = raw["choices"][0]["message"]
    message["reasoning_content"] = message.pop("reasoning")
    assert "forbidden" in llm_client.normalize_response(raw)["reasoning"]
    message["reasoning"] = "conflict"
    with pytest.raises(ValueError, match="Conflicting"):
        llm_client.normalize_response(raw)


@pytest.mark.parametrize("raw", [response(None), response(""), response("{}", finish="length"), {"choices": []}])
def test_missing_or_incomplete_final_rejected(raw):
    with pytest.raises(ValueError):
        llm_client.normalize_response(raw)


def test_smoke_gate_then_extend_and_never_truncate(harness):
    runner, calls, _ = harness
    archive = runner.BASE_REPLAY_DIR / "models/qwen3-8b-thinking/llm_metrics.csv"
    archive.parent.mkdir(parents=True)
    archive.write_text("historical bytes must never change\n")
    with pytest.raises(ValueError, match="Production requires"):
        storage.run(arguments(), runner)
    assert calls == []
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    assert (directory / "validation/llm_metrics.json").exists()
    storage.run(arguments("--replicates", "3"), runner)
    storage.run(arguments("--replicates", "1"), runner)
    assert len(calls) == 30
    saved = pd.read_csv(directory / "llm_metrics.csv")
    assert len(saved) == 30 and saved.groupby("replicate")["experiment"].nunique().eq(10).all()
    assert archive.read_text() == "historical bytes must never change\n"


def test_protocol_change_requires_new_smoke(harness):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    with pytest.raises(ValueError, match="Production requires"):
        storage.run(arguments("--max-output-tokens", "2048"), runner)
    assert len(directories(runner)) == 2 and len(calls) == 10


def test_interruption_resumes_prefix_and_request_seed(harness):
    runner, calls, control = harness
    control["fail_at"] = 2
    with pytest.raises(RuntimeError, match="interrupted"):
        storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    checkpoint = directory / "checkpoints/llm_metrics/replicate_000.json"
    assert len(json.loads(checkpoint.read_text())["rows"]) == 1
    failed_request = calls[1]
    storage.run(arguments("--smoke-test"), runner)
    assert calls[2] == failed_request
    assert len(calls) == 11 and len(pd.read_csv(directory / "llm_metrics.csv")) == 10


def test_received_answer_reused_after_checkpoint_interruption(harness):
    runner, calls, _ = harness
    interrupt = [True]
    def complete(row, feedback):
        replay.complete(row, feedback)
        if interrupt and interrupt.pop():
            raise RuntimeError("crash after received answer")
    runner.complete = complete
    with pytest.raises(RuntimeError, match="crash after"):
        storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 1
    storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 10  # The first inference was not repeated.


def test_invalid_ids_fail_without_evaluations_or_replacement(harness):
    runner, calls, control = harness
    control["invalid"] = True
    with pytest.raises(ValueError, match="did not complete"):
        storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    state = json.loads((directory / "checkpoints/llm_metrics/replicate_000.json").read_text())
    assert state["status"] == "failed" and state["rows"] == [] and len(calls) == 3
    assert len({b["seed"] for b in calls}) == 3
    assert not (directory / "validation/llm_metrics.json").exists()
    with pytest.raises(ValueError, match="did not complete"):
        storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 3


def test_gate_detects_journal_tampering(harness):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    path = next((directory / "llm_logs/llm_metrics/replicate_000").glob("*.json"))
    record = json.loads(path.read_text())
    record["latency_seconds"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Production requires"):
        storage.run(arguments("--replicates", "2"), runner)
    assert len(calls) == 10


def test_checkpoint_and_csv_mismatch_refused(harness):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    path = directory / "llm_metrics.csv"
    saved = pd.read_csv(path, keep_default_na=False)
    saved.loc[0, "val_accuracy"] = 0.99
    saved.to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="CSV feedback"):
        storage.run(arguments("--replicates", "2"), runner)
    assert path.read_bytes() == before and len(calls) == 10


def test_existing_prompt_and_observed_only_feedback(harness):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    expected, _, _ = replay.llm_search.build_prompt([], "metrics")
    assert calls[0]["messages"] == [dict(role="user", content=expected)]
    assert "Validation accuracy:" not in calls[0]["messages"][0]["content"]
    for body in calls:
        assert "test_accuracy" not in json.dumps(body) and "test_f1" not in json.dumps(body)
    directory = directories(runner)[0]
    rows = json.loads((directory / "checkpoints/llm_metrics/replicate_000.json").read_text())["rows"]
    expected, _, _ = replay.llm_search.build_prompt(rows[:1], "metrics")
    assert calls[1]["messages"][0]["content"] == expected
    for trial, body in enumerate(calls):
        expected, _, _ = replay.llm_search.build_prompt(rows[:trial], "metrics")
        assert body["messages"] == [dict(role="user", content=expected)]


def test_secondary_profile_is_isolated_and_smoke_validated(harness):
    runner, calls, _ = harness
    for extra in ([], ["--decoding-profile", "publisher-nonthinking"]):
        storage.run(arguments("--llm-model", "qwen3.8-27b", "--thinking", "off", "--smoke-test", *extra), runner)
    assert len(directories(runner)) == 2 and len(calls) == 20
    assert calls[0]["temperature"] == 1.0 and calls[10]["temperature"] == 0.7
    assert calls[0]["seed"] == calls[10]["seed"]  # Controlled seed schedule, separately labelled conditions.


def test_smoke_rejects_unknown_usage(harness, monkeypatch):
    runner, _, _ = harness
    real = FakeClient.complete
    def missing(self, body):
        result = real(self, body)
        result.pop("usage")
        return result
    monkeypatch.setattr(FakeClient, "complete", missing)
    with pytest.raises(ValueError, match="token usage"):
        storage.run(arguments("--smoke-test"), runner)


def test_primary_discovery_counts_failed_trajectories():
    rows = []
    for status, found, accuracy in [("completed", True, 0.8), ("completed", False, 0.6), ("failed", False, np.nan)]:
        rows.append(dict(series="s", status=status, found_best=found, found_top3=found, best_val_accuracy=accuracy,
                         training_gpu_hours=1.0, incumbent_test_accuracy=0.5, incumbent_test_f1=0.4,
                         **llm_analysis.request_costs([])))
    table = llm_analysis.primary_summary(pd.DataFrame(rows), {"s": 3}, 0.9).iloc[0]
    assert table["p_found_best_at_budget"] == pytest.approx(1 / 3)
    assert table["failure_rate"] == pytest.approx(1 / 3)
    assert table["mean_best_val_accuracy_observed"] == pytest.approx(0.7)
    assert np.isnan(table["mean_best_val_accuracy"])
    unfinished = llm_analysis.primary_summary(pd.DataFrame(rows), {"s": 4}, 0.9).iloc[0]
    assert not unfinished["is_final"] and np.isnan(unfinished["p_found_best_at_budget"])


def test_analysis_is_offline_and_does_not_modify_sources(harness, monkeypatch, capsys):
    runner, _, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline analysis must not create an inference client")
    monkeypatch.setattr(llm_client, "LocalClient", forbidden)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "analyze", lambda *args: None)
    args = arguments("--analyze-only", "--analysis-inputs", str(directory))
    llm_analysis.run(args, runner)
    assert all(p.read_bytes() == content for p, content in before.items())
    summary = pd.read_csv(next(runner.BASE_OUTPUT_DIR.glob("llm_comparisons/*/primary_outcomes.csv")))
    assert summary.loc[0, "is_final"] and summary.loc[0, "n_started"] == 1
    final = pd.read_csv(next(runner.BASE_OUTPUT_DIR.glob("llm_comparisons/*/final_summary.csv")))
    diagnostics = pd.read_csv(next(runner.BASE_OUTPUT_DIR.glob("llm_comparisons/*/inference_diagnostics.csv")))
    manifest = json.loads(next(runner.BASE_OUTPUT_DIR.glob("llm_comparisons/*/comparison_metadata.json")).read_text())
    assert manifest["reporting_source_hashes"] == llm_analysis.reporting_source_hashes(replay.SEARCH_DIR)
    assert manifest["sources"][0]["metadata"] == json.loads((directory / "metadata.json").read_text())
    assert final.loc[0, "model"] == "qwen3-8b-thinking" and final.loc[0, "method"] == "llm_metrics"
    assert final.loc[0, "trial"] == 10 and final.loc[0, "n_replicates"] == 1
    for displayed, original in [("mean_best_val_accuracy", "mean_best_val_accuracy"),
                                ("mean_regret", "mean_simple_regret"),
                                ("p_found_best", "p_found_best_at_budget"),
                                ("p_found_best_ci95_low", "p_found_best_ci95_low"),
                                ("p_found_top3", "p_found_top3_at_budget"),
                                ("p_found_top3_ci95_high", "p_found_top3_ci95_high"),
                                ("mean_cumulative_gpu_hours", "mean_training_gpu_hours"),
                                ("mean_incumbent_test_accuracy", "mean_incumbent_test_accuracy"),
                                ("mean_incumbent_test_f1", "mean_incumbent_test_f1")]:
        assert final.loc[0, displayed] == pytest.approx(summary.loc[0, original])
    assert np.isnan(final.loc[0, "ci95_low"])  # A single trajectory has no bootstrap CI.
    assert diagnostics.loc[0, "mean_prompt_tokens"] == 30000
    assert diagnostics.loc[0, "inference_requests"] == 10
    output = capsys.readouterr().out
    assert "LLM replay summary" in output and "Best val [95% CI]" in output and "Test F1" in output
    assert "Inference and trajectory diagnostics" in output and "Dup complete %" in output


def saved_classical_csvs(runner, counts, budget=2):
    """Write synthetic completed CSVs directly, without invoking any optimiser."""
    directory = runner.BASE_REPLAY_DIR / "baselines" / "saved"
    directory.mkdir(parents=True)
    (directory / ".writer.lock").touch()
    metadata = dict(schema_version=1, protocol="natural", landscape="full", budget=budget,
                    base_seed=1000, landscape_sha256=hashlib.sha256(runner.LANDSCAPES["full"].read_bytes()).hexdigest(),
                    sources={"classical_search.py": "saved-execution-source"}, packages={"numpy": "saved-version"})
    metadata["protocol_hash"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (directory / "metadata.json").write_text(json.dumps(metadata))
    feedback = runner.load_landscape("full")
    for method, count in counts.items():
        names = [build_experiment_name(config) for config in classical_search.catalogue(method)]
        rows = []
        for replicate in range(count):
            for trial in range(1, budget + 1):
                name = names[(replicate + trial - 1) % len(names)]
                rows.append(dict(method=method, replicate=replicate, seed=1000 + replicate, trial=trial,
                                 experiment=name, landscape="full",
                                 **{key: round(feedback[name][key], 6) for key in ("val_accuracy", "val_f1")}))
        pd.DataFrame(rows).to_csv(directory / f"{method}.csv", index=False)
    return directory


def test_consolidated_cli_reads_saved_1000_classical_and_50_qwen_separately(harness, monkeypatch, capsys):
    runner, _, _ = harness
    methods = ["rf_ei", "tpe_discrete", "gp_ei", "multimodal_diverse"]
    classical = saved_classical_csvs(runner, {method: 1000 for method in methods})
    prompts = ["llm_metrics", "llm_metrics_combined_gpu"]
    common = ["--budget", "2", "--methods", *prompts]
    storage.run(arguments(*common, "--smoke-test"), runner)
    storage.run(arguments(*common, "--replicates", "50"), runner)
    # Same model and prompt under different sampling must remain a separate series.
    storage.run(arguments("--budget", "2", "--temperature", "0.7", "--smoke-test"), runner)
    inputs = [classical, *directories(runner)]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for directory in inputs for path in directory.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline reporting must only load saved results, without execution or new execution provenance")

    for module, name in [(llm_client, "LocalClient"), (storage, "run"), (storage, "protocol"),
                         (llm_analysis.baseline_io, "run_method"), (llm_analysis.baseline_io, "protocol"),
                         (replay, "replay"), (replay, "run_method"), (replay.llm_search, "query_llm")]:
        monkeypatch.setattr(module, name, forbidden)
    for name in ("LANDSCAPES", "BASE_REPLAY_DIR", "BASE_OUTPUT_DIR", "load_landscape"):
        monkeypatch.setattr(replay, name, getattr(runner, name))
    monkeypatch.setattr(llm_analysis.baseline_analysis, "analyze", lambda *args: None)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "N_BOOTSTRAP", 40)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "COMPUTE_GRID_POINTS", 5)
    monkeypatch.setattr(sys, "argv", ["replay_search.py", "--analyze-only", "--methods", *methods, *prompts,
                                    "--landscape", "full", "--budget", "2", "--replicates", "1000",
                                    "--analysis-inputs", *map(str, inputs)])
    replay.main()
    output = next(runner.BASE_OUTPUT_DIR.glob("llm_comparisons/*"))
    final = pd.read_csv(output / "final_summary.csv")
    diagnostics = pd.read_csv(output / "inference_diagnostics.csv")
    coverage = pd.read_csv(output / "input_coverage.csv")
    manifest = json.loads((output / "comparison_metadata.json").read_text())
    assert len(final) == 7 and final["series"].is_unique and final["is_final"].all()
    assert final.loc[final["model"].eq("classical"), "n_replicates"].tolist() == [1000] * 4
    qwen = final.loc[final["model"].eq("qwen3-8b-thinking")]
    assert sorted(qwen["n_replicates"]) == [1, 50, 50]
    assert qwen["protocol_hash"].nunique() == 2
    assert qwen.loc[qwen["method"].eq("llm_metrics"), "series"].nunique() == 2
    assert diagnostics.loc[diagnostics["model"].eq("classical"), "inference_requests"].eq(0).all()
    assert sorted(diagnostics.loc[diagnostics["model"].eq("qwen3-8b-thinking"), "inference_requests"]) == [2, 100, 100]
    assert coverage["excluded_by_limit"].eq(0).all()
    assert {row["population_source"] for row in manifest["input_coverage"]} == {"completed_csv", "request_ledger"}
    assert len(manifest["sources"]) == 3
    assert set(manifest["sources"][0]["files"]) == {"metadata.json", *(f"{method}.csv" for method in methods)}
    # The consolidated endpoints use the same scoring definitions as classical reports.
    full = pd.read_csv(runner.LANDSCAPES["full"])
    for method in methods:
        saved = pd.read_csv(classical / f"{method}.csv").assign(series=method)
        scored, _, _, _ = llm_analysis.baseline_analysis.score_trials(saved, full)
        endpoints = scored.loc[scored["trial"].eq(2)]
        row = final.loc[final["method"].eq(method)].iloc[0]
        for displayed, metric in [("mean_best_val_accuracy", "best_so_far"), ("mean_regret", "regret"),
                                  ("p_found_best", "found_best"), ("p_found_top3", "found_top3"),
                                  ("mean_cumulative_gpu_hours", "cumulative_gpu_hours"),
                                  ("mean_incumbent_test_accuracy", "incumbent_test_accuracy"),
                                  ("mean_incumbent_test_f1", "incumbent_test_f1")]:
            assert row[displayed] == pytest.approx(endpoints[metric].mean())
        low, high = llm_analysis.baseline_analysis.wilson(endpoints["found_best"].sum(), 1000)
        assert row["p_found_best_ci95_low"] == pytest.approx(low)
        assert row["p_found_best_ci95_high"] == pytest.approx(high)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest()
             for directory in inputs for path in directory.rglob("*") if path.is_file()}
    assert after == before
    console = capsys.readouterr().out
    assert "Saved classical/rf_ei: including 1000 of 1000 completed CSV trajectories" in console
    assert "Saved qwen3-8b-thinking/llm_metrics: including 50 of 50 ledger-requested trajectories" in console


def test_offline_replicate_ceiling_reports_excluded_saved_classical_rows(harness, monkeypatch, capsys):
    runner, _, _ = harness
    directory = saved_classical_csvs(runner, {"rf_ei": 1000})
    before = (directory / "rf_ei.csv").read_bytes()
    monkeypatch.setattr(llm_analysis.baseline_analysis, "analyze", lambda *args: None)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "N_BOOTSTRAP", 40)
    output = llm_analysis.run(arguments("--analyze-only", "--methods", "rf_ei", "--budget", "2",
                                        "--replicates", "50", "--analysis-inputs", str(directory)), runner)
    coverage = pd.read_csv(output / "input_coverage.csv").iloc[0]
    assert coverage["saved_population"] == 1000 and coverage["included_replicates"] == 50
    assert coverage["excluded_by_limit"] == 950 and coverage["replicate_id_limit"] == 50
    assert (directory / "rf_ei.csv").read_bytes() == before
    assert "--replicates 50 excludes saved replicate IDs >= 50" in capsys.readouterr().out


def test_offline_comparison_refuses_missing_requested_method(harness):
    runner, _, _ = harness
    directory = saved_classical_csvs(runner, {"rf_ei": 1})
    with pytest.raises(ValueError, match="No saved input found for requested methods: gp_ei"):
        llm_analysis.run(arguments("--analyze-only", "--methods", "rf_ei", "gp_ei", "--budget", "2",
                                   "--analysis-inputs", str(directory)), runner)
    assert not runner.BASE_OUTPUT_DIR.exists()


def saved_historical_csvs(runner, model, methods=("llm_metrics",), count=10, budget=2):
    """The historical Responses format contains completed CSVs and text-only logs."""
    directory = runner.BASE_REPLAY_DIR / "models" / model
    directory.mkdir(parents=True)
    feedback = runner.load_landscape("full")
    names = list(feedback)
    for method in methods:
        rows = [dict(method=method, replicate=replicate, seed=1000 + replicate, trial=trial,
                     experiment=names[trial - 1], landscape="full", selection_reason="synthetic archive",
                     experiment_summary="", **{key: round(feedback[names[trial - 1]][key], 6)
                                               for key in ("val_accuracy", "val_f1")})
                for replicate in range(count) for trial in range(1, budget + 1)]
        pd.DataFrame(rows).to_csv(directory / f"{method}.csv", index=False)
        logs = directory / "llm_logs" / method
        logs.mkdir(parents=True)
        for replicate in range(count):
            (logs / f"replicate_{replicate:03d}.json").write_text(json.dumps([
                dict(type="decision", trial=trial, attempt=1, prompt="saved prompt", response="saved text")
                for trial in range(1, budget + 1)]))
    return directory


def test_historical_astra_opus_join_classical_and_qwen_without_fabricated_provenance(harness, monkeypatch, capsys):
    runner, _, _ = harness
    classical = saved_classical_csvs(runner, {"rf_ei": 3})
    prompts = list(runner.LLM_VARIANTS)
    archives = [saved_historical_csvs(runner, model, prompts)
                for model in ("openai_gpt6_astra", "anthropic_claude_opus_5_5")]
    storage.run(arguments("--budget", "2", "--smoke-test", "--methods", "llm_metrics", "llm_metrics_combined_gpu"), runner)
    qwen = next(path for path in directories(runner) if path.name.startswith("controlled_"))
    inputs = [classical, qwen, *archives]
    before = {path: path.read_bytes() for directory in inputs for path in directory.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical comparison must not run any experimental code")

    for module, name in [(llm_client, "LocalClient"), (storage, "run"), (replay, "replay"),
                         (llm_analysis.baseline_io, "run_method")]:
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "analyze", lambda *args: None)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "N_BOOTSTRAP", 40)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "COMPUTE_GRID_POINTS", 5)
    output = llm_analysis.run(arguments("--analyze-only", "--methods", "rf_ei", *prompts,
                                        "--budget", "2", "--replicates", "1000",
                                        "--analysis-inputs", *map(str, inputs)), runner)
    final = pd.read_csv(output / "final_summary.csv")
    historical = final.loc[final["input_kind"].eq("historical_csv")]
    assert len(final) == 3 + 2 * len(prompts) and final["series"].is_unique
    assert historical.groupby("model")["method"].agg(set).tolist() == [set(prompts), set(prompts)]
    assert historical["n_replicates"].eq(10).all() and historical["n_completed"].eq(10).all()
    assert historical["protocol_hash"].isna().all() and historical["snapshot_sha256"].str.len().eq(64).all()
    assert historical["thinking"].eq("unrecorded").all() and historical["failure_accounting"].eq("unavailable").all()
    assert historical["population_scope"].str.contains("saved completed trajectories only").all()
    assert historical[["n_failed", "n_in_progress", "n_not_started"]].isna().all().all()
    assert historical["mean_best_val_accuracy"].notna().all() and historical["ci95_low"].notna().all()
    assert final.loc[final["model"].eq("qwen3-8b-thinking"), "protocol_hash"].str.len().eq(64).all()
    diagnostics = pd.read_csv(output / "inference_diagnostics.csv")
    old = diagnostics.loc[diagnostics["input_kind"].eq("historical_csv")]
    unknown = ["inference_requests", "mean_prompt_tokens", "mean_completion_tokens", "mean_reasoning_tokens",
               "mean_inference_seconds", "failure_rate", "invalid_outputs", "transport_errors", "missing_usage"]
    assert old[unknown].isna().all().all()
    assert old["completed_duplicate_trajectory_rate"].eq(0.9).all()
    curves = pd.read_csv(output / "all_trajectory_curves.csv")
    assert curves.loc[curves["series"].isin(historical["series"]), "population_scope"].str.contains("failure denominator unknown").all()
    manifest = json.loads((output / "comparison_metadata.json").read_text())
    for source in manifest["sources"][2:]:
        assert source["metadata"]["protocol_hash"] is None
        assert source["metadata"]["metadata_origin"] == "offline analysis; not recorded execution metadata"
        assert "llm_logs/llm_metrics/replicate_000.json" in source["files"]
    assert {path: path.read_bytes() for directory in inputs for path in directory.rglob("*") if path.is_file()} == before
    assert all(not (directory / name).exists() for directory in archives for name in ("metadata.json", ".writer.lock"))
    assert "Snapshot hashes identify archive contents, not verified execution protocols" in capsys.readouterr().out


def test_analysis_input_filter_selects_methods_per_protocol(harness, monkeypatch):
    runner, _, _ = harness
    methods = ["llm_metrics", "llm_metrics_combined_gpu"]
    storage.run(arguments("--budget", "2", "--methods", *methods, "--smoke-test"), runner)
    qwen = directories(runner)[0]
    astra = saved_historical_csvs(runner, "openai_gpt6_astra", methods)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "analyze", lambda *args: None)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "N_BOOTSTRAP", 40)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "COMPUTE_GRID_POINTS", 5)
    output = llm_analysis.run(arguments(
        "--analyze-only", "--budget", "2", "--methods", *methods,
        "--analysis-inputs", f"{qwen}::llm_metrics", str(astra)), runner)
    final = pd.read_csv(output / "final_summary.csv")
    assert set(zip(final["model"], final["method"])) == {
        ("qwen3-8b-thinking", "llm_metrics"),
        ("openai_gpt6_astra", "llm_metrics"),
        ("openai_gpt6_astra", "llm_metrics_combined_gpu"),
    }
    manifest = json.loads((output / "comparison_metadata.json").read_text())
    assert manifest["sources"][0]["selected_methods"] == ["llm_metrics"]
    assert manifest["sources"][1]["selected_methods"] == methods
    assert not any("llm_metrics_combined_gpu" in name for name in manifest["sources"][0]["files"])


def test_analysis_input_filter_rejects_ambiguous_or_unrequested_methods(tmp_path):
    with pytest.raises(ValueError, match="also appear in --methods"):
        llm_analysis.input_spec(f"{tmp_path}::llm_summary", ["llm_metrics"])
    with pytest.raises(ValueError, match="PATH::method"):
        llm_analysis.input_spec(f"{tmp_path}::", ["llm_metrics"])


def test_reporting_tolerates_only_adjacent_cent_observed_cost_rendering(harness):
    runner, _, _ = harness
    feedback = runner.load_landscape("full")
    for values in feedback.values():
        values["runtime_hours"] = 1.005
    storage.run(arguments("--budget", "2", "--methods", "llm_metrics_combined_gpu", "--smoke-test"), runner)
    directory = directories(runner)[0]
    metadata = json.loads((directory / "metadata.json").read_text())
    path = directory / "llm_logs/llm_metrics_combined_gpu/replicate_000/trial_002_decision_attempt_01_transport_0000.json"
    record = json.loads(path.read_text())
    prompt = record["request"]["messages"][0]["content"]
    match = re.search(r"GPU hours used so far: ([0-9]+\.[0-9]{2})", prompt)
    assert match
    adjacent = f"{float(match.group(1)) + 0.01:.2f}"
    prompt = prompt[:match.start(1)] + adjacent + prompt[match.end(1):]
    record["request"]["messages"][0]["content"] = prompt
    record["prompt_sha256"] = llm_client.digest(prompt)
    storage.atomic_json(path, record)
    with pytest.raises(ValueError, match="observed-history"):
        storage.validate_saved_method(directory, metadata, "llm_metrics_combined_gpu", feedback)
    assert llm_analysis.validate_saved_method_for_analysis(
        directory, metadata, "llm_metrics_combined_gpu", feedback) == 1

    prompt = prompt.replace("Balance exploration and exploitation.", "Balance exploration and exploitation. Tampered.")
    record["request"]["messages"][0]["content"] = prompt
    record["prompt_sha256"] = llm_client.digest(prompt)
    storage.atomic_json(path, record)
    with pytest.raises(ValueError, match="observed-history"):
        llm_analysis.validate_saved_method_for_analysis(
            directory, metadata, "llm_metrics_combined_gpu", feedback)


@pytest.mark.parametrize("damage,match", [("partial", "incomplete"), ("duplicate", "incomplete"),
                                         ("feedback", "feedback"), ("proxy", "landscape")])
def test_historical_archive_refuses_invalid_rows_without_filtering(harness, damage, match):
    runner, _, _ = harness
    directory = saved_historical_csvs(runner, "openai_gpt6_astra")
    path = directory / "llm_metrics.csv"
    table = pd.read_csv(path)
    if damage == "partial":
        table = table.iloc[:-1]
    elif damage == "duplicate":
        table.loc[1, "experiment"] = table.loc[0, "experiment"]
    elif damage == "feedback":
        table.loc[0, "val_accuracy"] += 0.01
    else:
        table.loc[0, "landscape"] = "proxy"
    table.to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=match):
        llm_analysis.run(arguments("--analyze-only", "--budget", "2", "--analysis-inputs", str(directory)), runner)
    assert path.read_bytes() == before and not runner.BASE_OUTPUT_DIR.exists()


def test_historical_snapshot_change_refuses_report(harness, monkeypatch):
    runner, _, _ = harness
    directory = saved_historical_csvs(runner, "anthropic_claude_opus_5_5")
    path = directory / "llm_metrics.csv"
    validate = llm_analysis.baseline_io.validate_trials
    def changing(*args, **kwargs):
        result = validate(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")  # Simulate a non-cooperating historical writer.
        return result
    monkeypatch.setattr(llm_analysis.baseline_io, "validate_trials", changing)
    with pytest.raises(ValueError, match="Historical inputs changed during analysis"):
        llm_analysis.run(arguments("--analyze-only", "--budget", "2", "--analysis-inputs", str(directory)), runner)
    assert not runner.BASE_OUTPUT_DIR.exists()


@pytest.mark.parametrize("model,marker,match", [
    ("qwen3-8b-thinking", None, "Missing protocol lock"),
    ("openai_gpt6_astra", "checkpoints", "provenance downgrade"),
    ("openai_gpt6_astra", "metadata.json", "Missing protocol lock"),
])
def test_historical_adapter_cannot_bypass_modern_metadata(harness, model, marker, match):
    runner, _, _ = harness
    directory = saved_historical_csvs(runner, model)
    if marker:
        (directory / marker).touch()
    with pytest.raises(ValueError, match=match):
        llm_analysis.run(arguments("--analyze-only", "--budget", "2", "--analysis-inputs", str(directory)), runner)
    assert not runner.BASE_OUTPUT_DIR.exists()


def test_existing_remote_endpoint_allowed_without_connection(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://external.invalid/v1")
    client = llm_client.LocalClient(llm_client.settings(arguments()))
    assert client.base == "https://external.invalid/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://key:secret@external.invalid/v1")
    with pytest.raises(ValueError, match="credentials"):
        llm_client.LocalClient(llm_client.settings(arguments()))


def test_preflight_proves_control_rendering_when_available(monkeypatch):
    config = llm_client.settings(arguments())
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    client = llm_client.LocalClient(config)
    ignore_controls = [False]
    def http(path, body=None, **kwargs):
        if path == "/version": return {"version": "different-compatible-version"}
        if path == "/v1/models": return {"data": [{"id": config["model"]}]}
        if path.startswith("/server_info"):
            return dict(vllm_config=dict(model_config=dict(model=config["model_id"], revision=config["revision"], max_model_len=32768,
                                                           hf_token="private-token-must-not-be-recorded"),
                                         scheduler_config=dict(max_num_seqs=1), parallel_config={}, cache_config={}),
                        system_env=dict(nvidia_gpu_models="H200", nvidia_driver_version="synthetic"))
        if path == "/tokenizer_info": return dict(tokenizer_class="synthetic", chat_template="synthetic template")
        if path == "/openapi.json": raise HTTPError(path, 404, "optional", {}, None)
        if path == "/v1/chat/completions/render":
            value = 1 if ignore_controls[0] or body["chat_template_kwargs"]["enable_thinking"] else 2
            sampling = {k: body[k] for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
                                             "repetition_penalty", "min_p", "seed", "n")}
            sampling["max_tokens"] = body["max_tokens"]
            return dict(token_ids=[value], sampling_params=sampling)
        if path == "/detokenize": return {"prompt": str(body["tokens"])}
        raise AssertionError("Preflight must not run inference")
    monkeypatch.setattr(client, "http", http)
    facts = client.preflight()["provenance"]
    assert facts["reasoning_control_rendered"]["status"] == "verified"
    assert facts["server_version"]["value"] == "different-compatible-version"
    assert "private-token-must-not-be-recorded" not in json.dumps(facts)
    ignore_controls[0] = True
    with pytest.raises(ValueError, match="did not change"):
        client.preflight()


@pytest.mark.parametrize("field", ["seed", "max_tokens", "temperature"])
def test_render_rejects_engine_setting_overrides(monkeypatch, field):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    config = llm_client.settings(arguments())
    client = llm_client.LocalClient(config)
    sampling = {k: config[k] for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
                                      "repetition_penalty", "min_p", "max_tokens")}
    sampling.update(seed=7, n=1)
    sampling[field] += 1
    monkeypatch.setattr(client, "http", lambda *args, **kwargs: dict(token_ids=[1], sampling_params=sampling))
    with pytest.raises(ValueError, match="Engine changed"):
        client.render("prompt", seed=7)


def test_context_cap_never_shrinks(harness, monkeypatch):
    runner, calls, _ = harness
    real = FakeClient.render
    def large(self, prompt, config=None, seed=0):
        result = real(self, prompt, config, seed)
        result["count"] = 32768
        return result
    monkeypatch.setattr(FakeClient, "render", large)
    with pytest.raises(ValueError, match="headroom"):
        storage.run(arguments("--smoke-test"), runner)
    assert not calls
    state = json.loads((directories(runner)[0] / "checkpoints/llm_metrics/replicate_000.json").read_text())
    assert state["status"] == "in_progress" and not state["rows"]


def test_duplicate_selection_does_not_consume_budget(harness, monkeypatch):
    runner, calls, _ = harness
    real = FakeClient.complete
    def duplicate(self, body):
        result = real(self, body)
        result["choices"][0]["message"]["content"] = json.dumps(dict(configuration_id="config_001", reason="duplicate"))
        return result
    monkeypatch.setattr(FakeClient, "complete", duplicate)
    with pytest.raises(ValueError, match="did not complete"):
        storage.run(arguments("--smoke-test"), runner)
    state = json.loads((directories(runner)[0] / "checkpoints/llm_metrics/replicate_000.json").read_text())
    assert state["status"] == "failed" and len(state["rows"]) == 1 and len(calls) == 4


def test_thinking_off_rejects_reasoning_output(harness, monkeypatch):
    runner, _, _ = harness
    real = FakeClient.complete
    def unwanted(self, body):
        result = real(self, body)
        result["choices"][0]["message"]["reasoning"] = "Unexpected reasoning"
        return result
    monkeypatch.setattr(FakeClient, "complete", unwanted)
    with pytest.raises(ValueError, match="did not complete"):
        storage.run(arguments("--thinking", "off", "--smoke-test"), runner)
    state = json.loads((directories(runner)[0] / "checkpoints/llm_metrics/replicate_000.json").read_text())
    assert state["status"] == "failed" and not state["rows"]


def test_usage_mismatch_blocks_smoke(harness, monkeypatch):
    runner, _, _ = harness
    real = FakeClient.complete
    def mismatch(self, body):
        result = real(self, body)
        result["usage"]["prompt_tokens"] += 1
        return result
    monkeypatch.setattr(FakeClient, "complete", mismatch)
    with pytest.raises(ValueError, match="Actual prompt usage"):
        storage.run(arguments("--smoke-test"), runner)


def test_invalid_telemetry_stays_unknown_in_analysis():
    costs = llm_analysis.request_costs([
        dict(status="invalid_output", latency_seconds=3.0, response=dict(choices=[None], usage="bad")),
        dict(status="pending"),
    ])
    assert costs["invalid_outputs"] == 1 and costs["pending_requests"] == 1
    assert costs["missing_usage"] == 2 and costs["missing_latency"] == 1
    assert costs["inference_seconds"] == 3.0 and costs["inference_request_hours"] == 3 / 3600
    assert np.isnan(costs["inference_request_gpu_hours_estimate"])


def test_pending_summary_resumes_without_reselecting(harness):
    runner, calls, control = harness
    control["fail_at"] = 2  # The first summary, after one evaluated configuration.
    with pytest.raises(RuntimeError, match="interrupted"):
        storage.run(arguments("--methods", "llm_summary", "--smoke-test"), runner)
    directory = directories(runner)[0]
    checkpoint = directory / "checkpoints/llm_summary/replicate_000.json"
    rows = json.loads(checkpoint.read_text())["rows"]
    assert len(rows) == 1 and not rows[0]["experiment_summary"]
    storage.run(arguments("--methods", "llm_summary", "--smoke-test"), runner)
    rows = json.loads(checkpoint.read_text())["rows"]
    assert len(rows) == 10 and all(row["experiment_summary"] for row in rows)
    assert len(calls) == 21 and calls[1] == calls[2]
    assert calls[1]["seed"] != calls[0]["seed"]


def test_combined_gpu_uses_original_prompt(harness):
    runner, calls, _ = harness
    storage.run(arguments("--methods", "llm_metrics_combined_gpu", "--smoke-test"), runner)
    directory = directories(runner)[0]
    rows = json.loads((directory / "checkpoints/llm_metrics_combined_gpu/replicate_000.json").read_text())["rows"]
    for trial, body in enumerate(calls):
        expected, _, _ = replay.llm_search.build_prompt(rows[:trial], "metrics", shuffle_seed=2000,
                                                       budget=10, isolation_principles=False, gpu_hours_aware=True)
        assert body["messages"][0]["content"] == expected


def test_standard_api_only_smoke_and_resume(harness, monkeypatch):
    runner, calls, control = harness
    routes = []
    monkeypatch.setenv("OPENAI_BASE_URL", "http://existing.invalid:8123/v1")
    def http(self, path, body=None, **kwargs):
        routes.append(path)
        if path == "/v1/models":
            return dict(data=[dict(id=self.config["model"], root=self.config["model_id"], max_model_len=16384)])
        if path == "/v1/chat/completions":
            return FakeClient(self.config, calls, control).complete(body)
        raise HTTPError(path, 404, "No optional routes", {}, None)
    monkeypatch.setattr(HTTPClient, "http", http)
    monkeypatch.setattr(llm_client, "LocalClient", HTTPClient)
    storage.run(arguments("--smoke-test"), runner)
    storage.run(arguments("--replicates", "2"), runner)
    directory = directories(runner)[0]
    assert len(calls) == 20
    metadata = json.loads((directory / "metadata.json").read_text())
    facts = metadata["server"]["provenance"]
    assert metadata["server"]["endpoint"] == "http://existing.invalid:8123/v1"
    assert facts["served_model"]["status"] == "verified"
    assert facts["revision"]["status"] == "unavailable" and facts["revision"]["value"] is None
    assert facts["context_window"]["status"] == "verified"
    assert facts["effective_sampling"]["status"] == "unavailable"
    gate = json.loads((directory / "validation/llm_metrics.json").read_text())
    assert gate["verification"]["effective_sampling"] == "unavailable"
    assert gate["verification"]["reasoning_control"] == "unavailable"
    assert gate["limitations"]
    assert routes.count("/v1/chat/completions") == 20
    assert all("max_tokens" in body and "max_completion_tokens" not in body for body in calls)


def test_endpoint_change_cannot_resume(harness, monkeypatch):
    runner, calls, _ = harness
    monkeypatch.setenv("OPENAI_BASE_URL", "http://one.invalid/v1")
    storage.run(arguments("--smoke-test"), runner)
    original = directories(runner)[0] / "checkpoints/llm_metrics/replicate_000.json"
    before = original.read_bytes()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://two.invalid/v1")
    with pytest.raises(ValueError, match="server.endpoint"):
        storage.run(arguments("--replicates", "2"), runner)
    assert len(calls) == 10 and original.read_bytes() == before
    assert len(directories(runner)) == 1


def test_missing_diagnostics_do_not_invent_revision(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://existing.invalid/v1")
    config = llm_client.settings(arguments("--model-revision", "a" * 40, "--context-window", "16384"))
    client = HTTPClient(config)
    def http(path, body=None, **kwargs):
        if path == "/v1/models": return dict(data=[dict(id=config["model"])])
        raise HTTPError(path, 403, "Diagnostic access disabled", {}, None)
    monkeypatch.setattr(client, "http", http)
    facts = client.preflight()["provenance"]
    assert facts["revision"] == llm_client.evidence("a" * 40, "--model-revision", "user_supplied")
    assert facts["context_window"]["status"] == "user_supplied"
    assert facts["server_version"]["status"] == "unavailable"
    assert facts["tokenizer_class"]["status"] == "unavailable"
    assert llm_client.settings(arguments())["revision"] is None


def test_wrong_served_alias_blocks_before_inference(monkeypatch):
    client = HTTPClient(llm_client.settings(arguments()))
    calls = []
    def http(path, body=None, **kwargs):
        calls.append(path)
        return dict(data=[dict(id="wrong-model")])
    monkeypatch.setattr(client, "http", http)
    with pytest.raises(ValueError, match="served-model alias"):
        client.preflight()
    assert calls == ["/v1/models"]


@pytest.mark.parametrize("option,value,reported,match", [
    ("--model-revision", "a" * 40, dict(revision="b" * 40), "revision differs"),
    ("--context-window", "16384", dict(max_model_len=8192), "context length differs"),
])
def test_server_assertion_conflicts_fail(monkeypatch, option, value, reported, match):
    config = llm_client.settings(arguments(option, value))
    client = HTTPClient(config)
    def http(path, body=None, **kwargs):
        if path == "/v1/models": return dict(data=[dict(id=config["model"], **reported)])
        raise HTTPError(path, 404, "optional", {}, None)
    monkeypatch.setattr(client, "http", http)
    with pytest.raises(ValueError, match=match):
        client.preflight()


def test_existing_16k_context_requires_explicit_smaller_output_cap(monkeypatch):
    config = llm_client.settings(arguments("--max-output-tokens", "16384"))
    client = HTTPClient(config)
    def http(path, body=None, **kwargs):
        if path == "/v1/models": return dict(data=[dict(id=config["model"], max_model_len=16384)])
        raise HTTPError(path, 404, "optional", {}, None)
    monkeypatch.setattr(client, "http", http)
    with pytest.raises(ValueError, match="no server restart"):
        client.preflight()
    assert config["max_tokens"] == 16384


def test_schema_evidence_is_not_effective_settings(monkeypatch):
    config = llm_client.settings(arguments())
    client = HTTPClient(config)
    fields = {k: {} for k in llm_client.request_body("probe", config, 0)}
    schema = dict(paths={"/v1/chat/completions": dict(post=dict(requestBody=dict(content={
        "application/json": dict(schema={"$ref": "#/components/schemas/Chat"})})))},
        components=dict(schemas=dict(Chat=dict(properties=fields))))
    def http(path, body=None, **kwargs):
        if path == "/v1/models": return dict(data=[dict(id=config["model"], max_model_len=16384)])
        if path == "/openapi.json": return schema
        raise HTTPError(path, 404, "optional", {}, None)
    monkeypatch.setattr(client, "http", http)
    facts = client.preflight()["provenance"]
    assert facts["request_fields_declared"]["status"] == "verified"
    assert all(facts["request_fields_declared"]["value"].values())
    assert facts["effective_sampling"]["status"] == "unavailable"


def test_unsupported_chat_parameter_is_not_removed_and_retried(harness, monkeypatch):
    runner, calls, _ = harness
    def unsupported(self, body):
        calls.append(body)
        raise HTTPError("http://existing.invalid/v1/chat/completions", 400, "unsupported top_k", {}, None)
    monkeypatch.setattr(FakeClient, "complete", unsupported)
    with pytest.raises(RuntimeError, match="no parameters were dropped"):
        storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 1 and calls[0]["top_k"] == 20


def test_declared_unsupported_effort_is_rejected():
    schema = dict(paths={"/v1/chat/completions": dict(post=dict(requestBody=dict(content={
        "application/json": dict(schema=dict(properties=dict(reasoning_effort=dict(enum=["low", "medium"]))))})))})
    with pytest.raises(ValueError, match="does not support requested reasoning_effort"):
        llm_client.chat_properties(schema, dict(reasoning_effort="high"))


def test_unknown_context_and_mutable_revision_are_labelled(monkeypatch):
    config = llm_client.settings(arguments())
    client = HTTPClient(config)
    def http(path, body=None, **kwargs):
        if path == "/v1/models": return dict(data=[dict(id=config["model"], revision="main")])
        raise HTTPError(path, 404, "optional", {}, None)
    monkeypatch.setattr(client, "http", http)
    server = client.preflight()
    assert server["model"]["max_model_len"] is None
    assert server["provenance"]["context_window"]["status"] == "unavailable"
    assert server["provenance"]["reported_revision"]["value"] == "main"
    assert server["provenance"]["revision"]["status"] == "unavailable"
    assert any("immutable commit" in warning for warning in server["warnings"])


def test_unknown_gpu_count_is_not_assumed():
    records = [dict(status="accepted", latency_seconds=3600, response=response("{}"))]
    unknown = llm_analysis.request_costs(records)
    assert unknown["inference_request_hours"] == 1
    assert np.isnan(unknown["inference_request_gpu_hours_estimate"])
    assert llm_analysis.request_costs(records, gpu_count=2)["inference_request_gpu_hours_estimate"] == 2


def test_defaults_and_explicit_sampling_reuse_completed_smoke(harness, capsys):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    immutable = [*directory.glob("checkpoints/*/*.json"), *directory.glob("llm_logs/*/*/*.json"),
                 directory / "metadata.json", directory / "validation/llm_metrics.json"]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}
    storage.run(arguments("--smoke-test", "--temperature", "0.6", "--top-p", "0.95",
                          "--top-k", "20", "--presence-penalty", "0"), runner)
    assert len(calls) == 10 and directories(runner) == [directory]
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())
    assert "reused (completed; no inference sent)" in capsys.readouterr().out


def test_readable_layout_and_condition_names():
    args = arguments("--max-output-tokens", "8192")
    config = llm_client.settings(args)
    assert str(storage.readable_path(args, config)) == (
        "qwen3-8b-thinking/full/budget_010/seed_2000/thinking_on/"
        "controlled_t0.6_p0.95_k20_pp0_fp0_rp1_minp0_out8192")
    for flag, value in [("--temperature", "0.7"), ("--top-p", "0.9"), ("--top-k", "30"),
                        ("--presence-penalty", "0.5"), ("--thinking", "off"), ("--max-output-tokens", "4096")]:
        other = arguments("--max-output-tokens", "8192", flag, value)
        assert storage.readable_path(other, llm_client.settings(other)) != storage.readable_path(args, config)
    supplied = arguments("--llm-model", "Qwen/Qwen3-8B", "--condition", "revision-a")
    path = storage.readable_path(supplied, llm_client.settings(supplied))
    assert path.parts[0] == "Qwen%2FQwen3-8B" and path.name == "condition_revision-a"
    for name in ["../outside", "/absolute", "has spaces", "a" * 65]:
        with pytest.raises(ValueError, match="condition"):
            llm_client.settings(arguments("--condition", name))


@pytest.mark.parametrize("option,value,field", [
    ("--model-revision", "a" * 40, "generation.revision"),
    ("--context-window", "32768", "generation.context_window"),
])
def test_invisible_setting_collision_fails_before_inference(harness, option, value, field):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    before = {p: p.read_bytes() for p in directories(runner)[0].rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=field):
        storage.run(arguments("--smoke-test", option, value), runner)
    assert len(calls) == 10 and all(p.read_bytes() == data for p, data in before.items())
    storage.run(arguments("--smoke-test", option, value, "--condition", "asserted-checkpoint"), runner)
    assert len(calls) == 20 and len(directories(runner)) == 2


def test_changed_source_hash_refuses_readable_path(harness, monkeypatch):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    original = storage.protocol
    def changed(*args, **kwargs):
        path, metadata = original(*args, **kwargs)
        metadata["source_hashes"]["llm_client.py"] = "synthetic changed source"
        metadata["protocol_hash"] = llm_client.digest({k: v for k, v in metadata.items() if k != "protocol_hash"})
        return path, metadata
    monkeypatch.setattr(storage, "protocol", changed)
    with pytest.raises(ValueError, match="source_hashes.llm_client.py"):
        storage.run(arguments("--replicates", "2"), runner)
    assert len(calls) == 10


def test_extend_to_fifty_preserves_first_and_decreasing_preserves_all(harness):
    runner, calls, _ = harness
    # Short synthetic trajectories exercise the exact production replicate loop.
    storage.run(arguments("--smoke-test", "--budget", "1"), runner)
    directory = directories(runner)[0]
    first = (directory / "checkpoints/llm_metrics/replicate_000.json").read_bytes()
    storage.run(arguments("--budget", "1", "--replicates", "50"), runner)
    storage.run(arguments("--budget", "1", "--replicates", "1"), runner)
    assert len(calls) == 50
    assert (directory / "checkpoints/llm_metrics/replicate_000.json").read_bytes() == first
    assert json.loads((directory / "requests/llm_metrics.json").read_text())["replicates"] == 50
    assert len(pd.read_csv(directory / "llm_metrics.csv")) == 50
    assert len(list((directory / "checkpoints/llm_metrics").glob("*.json"))) == 50


@pytest.mark.parametrize("damage", ["metadata", "checkpoint", "checksum", "logs", "ledger", "orphan-status"])
def test_corruption_and_orphans_never_regenerate(harness, damage):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    checkpoint = directory / "checkpoints/llm_metrics/replicate_000.json"
    if damage == "metadata":
        (directory / "metadata.json").unlink()
    elif damage == "checkpoint":
        checkpoint.write_text('{"truncated":')
    elif damage == "checksum":
        state = json.loads(checkpoint.read_text())
        state["rows"][0]["selection_reason"] = "corrupt reason"
        checkpoint.write_text(json.dumps(state))
    elif damage == "logs":
        checkpoint.unlink()
        (directory / "llm_metrics.csv").unlink()
    elif damage == "ledger":
        (directory / "requests/llm_metrics.json").unlink()
    else:
        path = directory / "llm_metrics_status.csv"
        data = pd.read_csv(path)
        data.loc[0, "replicate"] = 5
        data.to_csv(path, index=False)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 10 and all(p.read_bytes() == data for p, data in before.items())


def test_changed_smoke_journal_cannot_be_revalidated(harness):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    path = next(directories(runner)[0].glob("llm_logs/*/*/*.json"))
    record = json.loads(path.read_text())
    record["latency_seconds"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="refusing to revalidate"):
        storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 10


def test_concurrent_llm_writers_refused(harness):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    metadata = json.loads((directory / "metadata.json").read_text())
    with storage.writer(directory, metadata):
        with pytest.raises(RuntimeError, match="Another replay"):
            storage.run(arguments("--replicates", "2"), runner)
    assert len(calls) == 10


def test_atomic_json_interruption_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    storage.atomic_json(path, {"value": "durable"})
    before = path.read_bytes()
    def interrupted(*args):
        raise OSError("synthetic rename interruption")
    monkeypatch.setattr(storage.os, "replace", interrupted)
    with pytest.raises(OSError, match="rename interruption"):
        storage.atomic_json(path, {"value": "new"})
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp"))


def test_seed_collision_stops_before_any_request(harness, monkeypatch):
    runner, calls, _ = harness
    monkeypatch.setattr(llm_client, "request_seed", lambda *args: 1)
    with pytest.raises(ValueError, match="seed collision"):
        storage.run(arguments("--smoke-test"), runner)
    assert calls == []


def test_full_production_seed_schedule_has_no_collisions():
    seeds = [llm_client.request_seed(2000, method, r, t, a)
             for method in ("llm_metrics", "llm_metrics_combined_gpu")
             for r in range(50) for t in range(1, 11) for a in range(1, 4)]
    assert len(seeds) == len(set(seeds)) == 3000
    assert all(0 <= seed < 2**32 for seed in seeds)


@pytest.mark.parametrize("raw", [[], None, {"choices": [None]}, {"choices": [dict(message="wrong")]},
                                  response("<think>{\"configuration_id\":\"config_001\"}</think>"),
                                  response("malformed JSON")])
def test_bad_final_answers_fail_without_random_fallback(harness, monkeypatch, raw):
    runner, calls, _ = harness
    def malformed(self, body):
        calls.append(body)
        return raw
    monkeypatch.setattr(FakeClient, "complete", malformed)
    with pytest.raises(ValueError, match="did not complete"):
        storage.run(arguments("--smoke-test"), runner)
    state = json.loads((directories(runner)[0] / "checkpoints/llm_metrics/replicate_000.json").read_text())
    assert state["status"] == "failed" and not state["rows"] and len(calls) == 3
    assert len({json.dumps({k: v for k, v in call.items() if k != "seed"}, sort_keys=True) for call in calls}) == 1


def test_unfinished_supporting_estimates_are_withheld():
    rows = [dict(series="s", status="in_progress", found_best=True, found_top3=True, best_val_accuracy=0.9,
                 training_gpu_hours=1, incumbent_test_accuracy=0.3, incumbent_test_f1=0.2,
                 **llm_analysis.request_costs([]))]
    result = llm_analysis.primary_summary(pd.DataFrame(rows), {"s": 50}, 0.9).iloc[0]
    assert result["n_not_started"] == 49 and result["p_found_best_observed_so_far"] == 1 / 50
    assert np.isnan(result["mean_best_val_accuracy"]) and np.isnan(result["best_val_accuracy_ci95_low"])
    assert np.isnan(result["failure_rate"])


def test_failed_prefixes_keep_denominator_at_budget_and_compute():
    full = pd.DataFrame(dict(experiment=["a", "b", "c"], val_accuracy=[0.9, 0.7, 0.5],
                             runtime_hours=[1, 1, 1], test_accuracy=[0.1, 0.8, 0.9], test_f1=[0.2, 0.7, 0.8]))
    trials = pd.DataFrame(dict(series=["s"] * 3, method=["llm_metrics"] * 3, model=["synthetic"] * 3,
                               replicate=[0, 0, 1], trial=[1, 2, 1], experiment=["a", "b", "c"], val_accuracy=[0.9, 0.7, 0.5]))
    endpoints = pd.DataFrame(dict(series=["s"] * 3, replicate=[0, 1, 2], status=["completed", "failed", "failed"]))
    curves, compute = llm_analysis.all_trajectory_curves([trials], endpoints, full, 2)
    assert curves["n_requested"].eq(3).all() and curves["p_found_best"].eq(1 / 3).all()
    assert curves["mean_best_val_accuracy"].isna().all()  # One failure has no incumbent.
    assert compute.iloc[-1]["p_found_best"] == 1 / 3
    assert curves.iloc[-1]["mean_training_gpu_hours"] == 1


def test_duplicate_trajectories_retained_in_offline_analysis(harness, monkeypatch):
    runner, _, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    storage.run(arguments("--replicates", "3"), runner)
    directory = directories(runner)[0]
    monkeypatch.setattr(llm_analysis.baseline_analysis, "analyze", lambda *args: None)
    llm_analysis.run(arguments("--analyze-only", "--replicates", "50", "--analysis-inputs", str(directory)), runner)
    diversity = pd.read_csv(next(runner.BASE_OUTPUT_DIR.glob("llm_comparisons/*/trajectory_diversity.csv")))
    assert diversity.loc[0, "n_completed"] == 3
    assert diversity.loc[0, "completed_duplicate_trajectory_rate"] == pytest.approx(2 / 3)
    assert len(json.loads(diversity.loc[0, "first_choice_frequencies"])) == 1


def test_transient_timeout_retries_same_payload_and_records_cost(harness, monkeypatch):
    runner, calls, _ = harness
    original = FakeClient.complete
    first = [True]
    def timeout(self, body):
        if first:
            first.pop()
            calls.append(body)
            raise TimeoutError("synthetic timeout")
        return original(self, body)
    monkeypatch.setattr(FakeClient, "complete", timeout)
    monkeypatch.setattr(storage.time, "sleep", lambda seconds: None)
    storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 11 and calls[0] == calls[1]
    records = [json.loads(p.read_text()) for p in directories(runner)[0].glob("llm_logs/*/*/*.json")]
    costs = llm_analysis.request_costs(records)
    assert costs["transport_errors"] == 1 and costs["missing_usage"] == 1


def test_response_system_fingerprint_change_stops_without_resampling(harness, monkeypatch):
    runner, calls, _ = harness
    original = FakeClient.complete
    def changed(self, body):
        raw = original(self, body)
        raw["system_fingerprint"] = "first" if len(calls) == 1 else "changed"
        return raw
    monkeypatch.setattr(FakeClient, "complete", changed)
    with pytest.raises(RuntimeError, match="system_fingerprint changed"):
        storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 2
    state = json.loads((directories(runner)[0] / "checkpoints/llm_metrics/replicate_000.json").read_text())
    assert state["status"] == "in_progress" and len(state["rows"]) == 1


def test_corrupt_saved_replicate_beyond_requested_limit_blocks(harness):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    storage.run(arguments("--replicates", "3"), runner)
    path = directories(runner)[0] / "checkpoints/llm_metrics/replicate_002.json"
    path.write_text("{interrupted file")
    with pytest.raises(ValueError, match="corrupt saved JSON"):
        storage.run(arguments("--replicates", "1"), runner)
    assert len(calls) == 30


def test_offline_analysis_reports_requested_but_not_started(harness, monkeypatch):
    runner, _, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    storage.atomic_json(directory / "requests/llm_metrics.json", dict(replicates=50))
    monkeypatch.setattr(llm_analysis.baseline_analysis, "analyze", lambda *args: None)
    llm_analysis.run(arguments("--analyze-only", "--replicates", "1000", "--analysis-inputs", str(directory)), runner)
    summary = pd.read_csv(next(runner.BASE_OUTPUT_DIR.glob("llm_comparisons/*/primary_outcomes.csv")))
    assert summary.loc[0, "n_started"] == 1 and summary.loc[0, "n_not_started"] == 49
    assert not summary.loc[0, "is_final"] and np.isnan(summary.loc[0, "mean_best_val_accuracy"])
    final = pd.read_csv(next(runner.BASE_OUTPUT_DIR.glob("llm_comparisons/*/final_summary.csv")))
    assert final.loc[0, "n_replicates"] == 50 and final.loc[0, "n_not_started"] == 49
    assert np.isnan(final.loc[0, "p_found_best"]) and np.isnan(final.loc[0, "mean_incumbent_test_accuracy"])


def test_copied_protocol_cannot_double_the_analysis_sample(harness, tmp_path, monkeypatch):
    import shutil
    runner, _, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    copied = tmp_path / "duplicate-input"
    shutil.copytree(directory, copied)
    monkeypatch.setattr(llm_analysis.baseline_analysis, "analyze", lambda *args: None)
    with pytest.raises(ValueError, match="Duplicate protocol/method input"):
        llm_analysis.run(arguments("--analyze-only", "--analysis-inputs", str(directory), str(copied)), runner)


def test_all_failed_curves_do_not_drop_zero_evaluation_failures():
    full = pd.DataFrame(dict(experiment=["a", "b", "c"], val_accuracy=[0.9, 0.7, 0.5]))
    endpoints = pd.DataFrame(dict(series=["s", "s"], replicate=[0, 1], status=["failed", "failed"]))
    curves, _ = llm_analysis.all_trajectory_curves([], endpoints, full, 10)
    assert len(curves) == 10 and curves["p_found_best"].eq(0).all()
    assert curves["n_requested"].eq(2).all() and curves["mean_best_val_accuracy"].isna().all()


@pytest.mark.parametrize("relative", ["checkpoints/llm_metrics/replicate_000.json",
                                      "llm_logs/llm_metrics/replicate_000/trial_001_decision_attempt_01_transport_0000.json"])
def test_stale_protocol_state_never_influences_another_condition(harness, relative):
    runner, calls, _ = harness
    storage.run(arguments("--smoke-test"), runner)
    original = directories(runner)[0]
    storage.run(arguments("--smoke-test", "--condition", "separate"), runner)
    other = next(path for path in directories(runner) if path != original)
    (other / relative).write_bytes((original / relative).read_bytes())
    with pytest.raises(ValueError, match="identity mismatch"):
        storage.run(arguments("--replicates", "2", "--condition", "separate"), runner)
    assert len(calls) == 20


def test_cli_automatically_analyzes_after_writer_releases_lock(harness, monkeypatch):
    runner, calls, _ = harness
    for name in ("BASE_REPLAY_DIR", "BASE_OUTPUT_DIR", "LANDSCAPES", "load_landscape"):
        monkeypatch.setattr(replay, name, getattr(runner, name))
    reports = []
    def analyze(args, module):
        directory = args.analysis_inputs[0]
        # Acquiring the shared reader lock proves inference's exclusive lock ended.
        with llm_analysis.reader(directory):
            for method in args.methods:
                state = storage.read_json(directory / "checkpoints" / method / "replicate_000.json")
                assert state["status"] == "completed" and len(state["rows"]) == 2
                assert (directory / "validation" / f"{method}.json").exists()
        assert args.analyze_only and args.replicates == 1
        assert module is replay
        reports.append(args)
    monkeypatch.setattr(llm_analysis, "run", analyze)
    argv = ["--llm-api", "chat", "--llm-model", "qwen3-8b-thinking",
            "--methods", "llm_metrics", "llm_metrics_combined_gpu", "--budget", "2",
            "--seed", "2000", "--max-output-tokens", "1024", "--replicates", "1", "--smoke-test"]
    children = []
    def child(command, **kwargs):
        assert command == [sys.executable, str(Path(replay_with_report.__file__).with_name("replay_search.py")), *argv]
        assert kwargs == {"check": True}  # No rewritten arguments, environment, cwd or streams.
        assert len(reports) == len(children)
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", command[1:])
            replay.main()  # Exercise the unchanged entry point with synthetic inference.
        children.append(command)
    monkeypatch.setattr(replay_with_report.subprocess, "run", child)
    replay_with_report.main(argv)
    assert len(calls) == 4 and len(reports) == 1
    replay_with_report.main(argv)  # Reusing completed smoke regenerates analysis, without inference.
    assert len(calls) == 4 and len(reports) == 2


def test_automatic_analysis_retains_saved_replicate_count(harness, monkeypatch):
    runner, calls, _ = harness
    args = arguments("--smoke-test")
    storage.run(args, runner)
    directory = directories(runner)[0]
    storage.atomic_json(directory / "requests/llm_metrics.json", dict(replicates=50))
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    reports = []
    def analyze(report_args, module):
        reports.append(report_args)
        assert report_args.replicates == 50 and report_args.analysis_inputs == [directory]
    def forbidden(*args, **kwargs):
        raise AssertionError("Automatic analysis must not create an inference client")
    monkeypatch.setattr(llm_analysis, "run", analyze)
    monkeypatch.setattr(llm_client, "LocalClient", forbidden)
    llm_analysis.after_replay(args, runner)
    assert len(reports) == 1 and len(calls) == 10
    assert args.replicates == 1 and args.analysis_inputs is None and not args.analyze_only
    assert all(path.read_bytes() == data for path, data in before.items())


def test_reporting_launcher_does_not_report_failed_execution(monkeypatch):
    def failed(command, **kwargs):
        raise replay_with_report.subprocess.CalledProcessError(7, command)
    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting must not run after failed execution")
    monkeypatch.setattr(replay_with_report.subprocess, "run", failed)
    monkeypatch.setattr(llm_analysis, "after_replay", forbidden)
    with pytest.raises(SystemExit) as error:
        replay_with_report.main(["--llm-api", "chat"])
    assert error.value.code == 7


def test_reporting_launcher_does_not_repeat_offline_analysis(monkeypatch, tmp_path):
    children = []
    def child(command, **kwargs):
        children.append(command)
    def forbidden(*args, **kwargs):
        raise AssertionError("The unchanged child entry point already handled --analyze-only")
    monkeypatch.setattr(replay_with_report.subprocess, "run", child)
    monkeypatch.setattr(llm_analysis, "after_replay", forbidden)
    argv = ["--llm-api", "chat", "--analyze-only", "--analysis-inputs", str(tmp_path)]
    replay_with_report.main(argv)
    assert len(children) == 1 and children[0][2:] == argv


@pytest.fixture
def copied_protocol_sources(harness, tmp_path):
    import shutil
    runner, calls, control = harness
    destination = tmp_path / "protocol_sources"
    destination.mkdir()
    for path in replay.SEARCH_DIR.glob("*.py"):
        shutil.copyfile(path, destination / path.name)
    runner.SEARCH_DIR = destination
    return runner, calls, control


@pytest.mark.parametrize("name", ["llm_analysis.py", "baseline_analysis.py", "replay_with_report.py"])
def test_reporting_only_source_changes_preserve_smoke_and_extension(copied_protocol_sources, name):
    runner, calls, _ = copied_protocol_sources
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    immutable = [directory / "metadata.json", directory / "validation/llm_metrics.json",
                 directory / "checkpoints/llm_metrics/replicate_000.json", *directory.glob("llm_logs/*/replicate_000/*.json")]
    before = {path: path.read_bytes() for path in immutable}
    original_reporting = llm_analysis.reporting_source_hashes(runner.SEARCH_DIR)
    source = runner.SEARCH_DIR / name
    source.write_text(source.read_text() + '\n# Revised offline table presentation.\n')
    revised_reporting = llm_analysis.reporting_source_hashes(runner.SEARCH_DIR)
    assert original_reporting[name] != revised_reporting[name]
    assert len(revised_reporting[name]) == 64  # Full audit hashes, independent of execution identity.
    storage.run(arguments("--smoke-test"), runner)
    assert len(calls) == 10
    storage.run(arguments("--replicates", "2"), runner)
    assert len(calls) == 20 and len(pd.read_csv(directory / "llm_metrics.csv")) == 20
    assert directories(runner) == [directory]
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("name,old,new", [
    ("llm_search.py", "Validation accuracy is the primary optimization objective.", "Validation F1 is the primary optimization objective."),
    ("llm_search.py", "random.Random(shuffle_seed).shuffle(configs)", "random.Random(shuffle_seed + 1).shuffle(configs)"),
    ("replay_search.py", 'val_accuracy=f"{result[\'val_accuracy\']:.6f}"', 'val_accuracy=f"{result[\'val_accuracy\']:.5f}"'),
    ("llm_client.py", '"llm-replay-request-v1"', '"different-request-seed-scheme"'),
    ("llm_client.py", "timeout_seconds=900", "timeout_seconds=901"),
    ("llm_client.py", 'text = message.get("content")', 'text = message.get("reasoning")'),
    ("llm_replay_io.py", 'range(1, config["selection_attempts"] + 1)', 'range(1, config["selection_attempts"] + 2)'),
])
def test_execution_source_changes_still_refuse_resume(copied_protocol_sources, name, old, new):
    runner, calls, _ = copied_protocol_sources
    storage.run(arguments("--smoke-test"), runner)
    directory = directories(runner)[0]
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    source = runner.SEARCH_DIR / name
    assert old in source.read_text()
    source.write_text(source.read_text().replace(old, new))
    with pytest.raises(ValueError, match="source_hashes"):
        storage.run(arguments("--replicates", "2"), runner)
    assert len(calls) == 10
    assert all(path.read_bytes() == data for path, data in before.items())
