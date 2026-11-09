# Local LLM replay audit — 2026-11-03

**Selective consolidated reporting follow-up — 2026-11-05.** Analysis inputs can
now use `PATH::method,method` to select methods per protocol without changing the
fingerprinted execution parser. This includes all seven Qwen3-8B, Astra and Opus
prompt variants while selecting only the completed `llm_metrics` arm from the
Qwen3.8-27B xhigh directory. Selected methods are recorded per source. Missing
global methods and duplicate input directories still fail closed.

Two saved Qwen3-8B `llm_metrics_combined_gpu` prompts contain `31.83` rather than
the `31.84` produced by rebuilding the same exact half-cent sum of nine observed
four-decimal runtimes. A reporting-only compatibility validator accepts only an
adjacent-cent difference on the single cumulative observed-cost line. Every
individual feedback value, prompt character outside that line, request setting,
seed, response checksum, selected configuration, checkpoint, export and failure
ledger remains strictly validated. Execution validation and source fingerprints
are unchanged. Synthetic tests cover the tolerated rendering and rejection of
any additional prompt change; tests were not executed.

**Historical proprietary comparison follow-up — 2026-11-04.** The consolidated
offline reader now accepts the explicitly selected original
`models/openai_gpt6_astra` and `models/anthropic_claude_opus_5_5` directories.
Each contains seven prompting conditions with ten completed budget-ten
trajectories. All original CSVs and text logs remain untouched. The report uses
the same scoring and intervals, identifies historical samples separately and
records archive-content hashes without inventing an execution fingerprint.
Historical failure/request denominators and inference telemetry are unavailable,
so their counts/rates/costs are `NA`; performance estimates describe the saved
completed sample. Model identity comes from the directory label, not a server
verification. Missing metadata elsewhere does not enable historical fallback.

The reader validates saved rows against the full landscape and rejects partial
or malformed trajectories. It reads historical data into a snapshot and refuses
changes during loading, without adding locks/metadata to the archives. Modern
locks, protocol fingerprints and checkpoint checks are unchanged. Synthetic
tests cover the combined classical/Qwen/Astra/Opus table, all historical prompt
variants, unknown telemetry, unchanged inputs, malformed/changing archives and
refusal to downgrade modern provenance. Only static checks are permitted; these
tests remain unexecuted. No execution source or scientific protocol changed.

**Saved-result reporting follow-up — 2026-11-04.** Read-only inspection now finds
1,000 completed ten-evaluation trajectories in each of the ten classical CSVs
under `baselines/full/budget_010_startup_003_seed_1000_pa8604c3c37c3`, and 50
completed ten-evaluation trajectories for each recent Qwen prompting condition
under the readable directory documented in [LLM_EXPERIMENTS.md](LLM_EXPERIMENTS.md#offline-analysis).
The consolidated loader reads those original CSVs and validated Qwen checkpoints
directly, without calling optimisers, inference or execution-protocol discovery.
There is no need to rerun the baseline experiments or generate intermediate
classical reports. The execution source hashes still match the saved Qwen
metadata; this reporting update needs no new smoke or experimental directory.

The local `search-full-landscape-replays` branch at `a857665` was inspected with
`git show`, without checkout or fetch. Its analysis also reads saved CSVs and
adds shared random/Bayesian results to model comparisons. The consolidated
analysis retains saved-result reuse, requires explicitly selected protocol
directories, includes the expanded classical method set and separates full LLM
protocol identities. Historical runs are not automatically pooled.

The shared offline replicate ceiling can accidentally restrict the classical
sample when set to 50. Reporting now prints saved/included counts and exclusions,
and saves `input_coverage.csv` plus matching manifest records. A requested method
absent from every input is rejected. The documented all-method comparison uses
`--replicates 1000`, preserving the current 1,000 classical / 50 Qwen counts.
Added synthetic tests cover that unequal-count combination through the actual
CLI, multiple Qwen sampling protocols, equivalence to classical scoring,
read-only inputs, forbidden generation paths, explicit truncation and missing
comparators. These tests were added but not executed.

**Reporting is now isolated from execution.** `replay_search.py` has been restored
byte for byte to the source recorded by the two successful human-readable-directory
smokes. The separate `replay_with_report.py` launcher forwards the original
arguments, environment and standard streams to that unchanged entry point in a
child process. After successful exit it runs offline analysis, prints the main
summary and inference/diversity diagnostics, and saves `final_summary.csv` and
`inference_diagnostics.csv`. No reporting-specific condition or directory is used.

All eight full execution-source hashes and the two recorded function hashes
match the existing readable-directory metadata. Its unchanged protocol hash is
`44298066f7e2952c5bbb26c8c083e44336b5ed2ddf4990a0653d3c98ca941cdd`.
Both checkpoints are completed with ten observations, and their checksums and
validation certificates remain intact. These smokes are eligible as replicate 0
when extending to 50, subject to the unchanged checks of generation settings,
environment and available server provenance at invocation. No checkpoint, journal,
certificate or metadata was edited or migrated. The server was not contacted.

The fingerprint algorithm and complete execution-source list are unchanged;
`replay_search.py` has not been excluded or normalized. Prompt, selection,
feedback, seed, parsing and retry changes still require compatible execution
metadata and otherwise stop before generation. Put offline reporting changes
in the launcher/analysis modules: their full SHA256 hashes are saved separately
under `reporting_source_hashes` in `comparison_metadata.json`, alongside the
original execution metadata. Reporting revisions change analysis provenance
without changing experimental compatibility. Execution modules remain
conservatively protected as complete source files.

Synthetic regression coverage includes unchanged argument forwarding, reporting
only after successful execution and lock release, no repeated offline report,
reuse/extension after reporting-source changes, and refusal after prompt,
selection, feedback, seed, parsing, timeout or retry source changes. Only static
checks were run; the added tests remain unexecuted.

The implementation has been reviewed and hardened on `search-extended-experiments`.
**Runtime validation is still required. No tests, inference, experiments, server
requests, installations, downloads or GPU jobs were run during this audit.**
Static syntax parsing, Ruff's Python correctness rules and whitespace checks
were used. Official Qwen documentation was consulted to confirm the request
interface; the running LLM server was not contacted.

The earlier audit introduced the approved request-seed scheme. The old
hash-directory smokes discussed below use its predecessor and remain separate.
That earlier incompatibility does not apply to the successful readable-directory
smokes above, which already use the current execution protocol. No automatic
migration or historical pooling is implemented.

## Findings and changes

| Severity | Finding | Resolution |
| --- | --- | --- |
| High | Some orphaned journals, missing request ledgers and checkpoints outside a reduced requested count could escape validation before generation. | Validate all saved method/replicate identities, checkpoint checksums, request seeds/settings, accepted decisions and CSV/status exports before generation. Reject orphaned or corrupt state without rewriting it. |
| High | Reissuing smoke validation could replace a saved validation certificate after evidence changed. | Existing certificates must match the unchanged checkpoint and journals; rerunning smoke cannot approve altered evidence. |
| High | Supporting means/intervals could be presented for unfinished studies; supplementary completion-only curves omit failures. | Withhold final means, discovery estimates and intervals until every requested trajectory is terminal. Add unconditional evaluation/compute curves retaining failures; label completion-only plots supplementary. |
| Medium | Repeated copies of the same protocol could be treated as additional independent analysis inputs. | Reject duplicate protocol/method inputs; retain naturally repeated sampled trajectories. |
| Medium | A truncated 32-bit seed hash had no collision check. | Use the approved `llm-replay-request-v1` namespace and check the complete requested/saved schedule before generation, including retry and summary seeds. Refuse collisions rather than silently changing seeds. |
| Medium | Metadata creation was not atomic and renamed directory entries were not explicitly flushed. | Use an LLM-specific locked writer, atomic metadata/checkpoints/journals, file fsync and parent-directory fsync. Classical storage is unchanged. |
| Medium | Hash-named result directories obscured settings. | Use readable settings-derived paths and retain full fingerprints in metadata. Field-level mismatch errors require an explicit separate condition; no automatic hash suffix. |
| Low | Completed smoke reuse printed only `completed`. | Print `reused (completed; no inference sent)`; terminal failures also explicitly report reuse without replacement. |
| Hardening | Malformed response objects and response-level server evidence needed clearer handling. | Validate response shapes, preserve final/reasoning separation, journal every attempt, and stop if a reported Chat `system_fingerprint` changes. |

The existing standard-API provenance design was sound: optional diagnostics are
not prerequisites, unavailable facts remain unavailable, user-supplied assertions
are distinct from server reports, and known contradictions stop the run. The
existing optional `--model-revision <40-character-commit>` is retained. Supply it
before smoke if desired and keep it unchanged through production. It does not
configure the server or independently verify its checkpoint.

## Evidence from the two earlier invocations

The preserved directory is:

```text
experiments/search/results/lookup_replays/models/qwen3-8b-thinking/full/budget_010_seed_2000_p695d776e3e3c
```

Its complete fingerprint is
`695d776e3e3c43abfbe811bb3c32cf35b737d71c62c40249db945335e1e6d0b0`.
Recomputing the fingerprint from saved metadata matched. Every source-file hash
recorded there matched the implementation at the start of this audit.

| Saved method | Checkpoint | Accepted requests | Other attempts | Request start times, UTC |
| --- | --- | ---: | ---: | --- |
| `llm_metrics` | replicate 0, completed, 10 observations | 10 | 0 | 01:48:14–01:50:17 |
| `llm_metrics_combined_gpu` | replicate 0, completed, 10 observations | 10 | 0 | 01:50:30–01:53:23 |

Both CSVs contain ten observations. Both validation certificates match the saved
checkpoint and journal hashes. All raw-response checksums match. Neither
checkpoint records an interruption. Requests consistently contain temperature
0.6, top-p 0.95, top-k 20, presence/frequency penalty 0.0, repetition penalty 1.0,
min-p 0.0, max_tokens 8192 and `enable_thinking: true`.

1. The first command's omitted values resolved to exactly the second command's
   explicit values. The identical fingerprint was correct; replicate count and
   selected method subset are intentionally outside protocol identity.
2. With these intact completed checkpoints, the old `run_replicate` returned
   before asking the journal or server for another response. The second command
   therefore reused the completed trajectories, rather than replaying individual
   responses or sampling another trajectory. Its status message was ambiguous.
3. The saved files contain only one trajectory per method and no evidence of
   regeneration. There is no invocation-level process log, so the files alone
   cannot independently reconstruct every action of the two shell invocations.
   This conclusion combines their intact evidence with the matching source code.
4. The old runner rewrote CSV exports and smoke certificates on reuse, but did
   not rewrite completed checkpoints or regenerate their selections. The new
   runner leaves completed checkpoints, request journals and existing smoke
   certificates unchanged; deterministic CSV/ledger exports may be refreshed.
5. A smoke is replicate 0 of its exact protocol. Increasing the total from 1 to
   50 extends IDs 1–49; it does not start another set of 50. Reissuing identical
   commands cannot replace a completed or terminally failed replicate.
6. Partial trajectories resume after their durable accepted prefix. A received
   answer saved before checkpoint commit is recovered from its exact journal
   request. An interrupted request whose response was never saved may be sent
   again with the same payload and seed; exact-once inference is not guaranteed.
7. Different metadata, requests or checkpoints are rejected before generation.
   Fresh readable paths do not search historical directories or import old logs.

The old metadata reports vLLM 0.33.0, endpoint `http://127.0.0.1:8000/v1`, served
alias `qwen3-8b-thinking`, model identifier `Qwen/Qwen3-8B` and context 16,384.
Immutable revision, effective sampling and rendered reasoning controls were
unavailable. Returned reasoning channels and context headroom were verified at
the API/output level. These facts do not prove the loaded weights or internal
seed/control application.

## Scientific and statistical invariants

- `llm_search.py`, `summarize_experiment.py` and `search_space.py` match both the
  saved smoke hashes and branch HEAD byte for byte. No prompt, catalogue, ID
  assignment, shuffling rule or selection objective was rewritten.
- The landscape contains 9 RNA-only, 6 imaging-only and 36 multimodal completed
  architectures. Its SHA256 remains
  `d2977505b24cea89180fd8198867d62c4728d2c0288544f0f8266a530e2f0289`.
- The global validation optimum remains `0.7249887585639954`, at
  `multimodal/multiprofile/train_transformerpool_on_scvi_n200_with_weighted_imagenet_vit_l`.
  Optimum and top-three sets, including ties, are computed exactly as in the
  established classical analysis. Full-resolution scoring is offline; prompts
  retain the historical six-decimal validation feedback.
- The request adapter sends one original user prompt. It adds no system/science
  instructions. Qwen's thinking switch is sent through
  `chat_template_kwargs.enable_thinking`, as documented by
  [Qwen](https://qwen.readthedocs.io/en/stable/deployment/vllm.html).
- The client sees the prompt and generation settings, not the landscape.
  Candidate descriptions contain no unobserved scores or measured unobserved
  costs. Prompt construction receives only accepted rows. `complete` looks up
  feedback only after selection. Held-out metrics are excluded from feedback;
  telemetry is logged locally, never appended to model messages.
- Strict JSON parsing uses final-answer `content` only. Reasoning fields are
  recorded separately. Missing/truncated/malformed outputs, invalid IDs and
  duplicates consume logged output attempts but no experimental evaluations.
  Exhaustion is a terminal failure; there is no random fallback, replacement
  sample, changed sampling setting or automatic token-cap increase.
- Transport failures are logged and leave an interrupted trajectory eligible
  for exact-protocol resumption. They do not disappear from coverage or cost
  accounting. Missing usage is reported as unknown, with known lower bounds.
- All requested IDs remain in reporting. A terminal failure can retain a hit
  achieved before it stopped, but cannot invent later evaluations. Failure
  before any accepted selection contributes a miss and no incumbent. Means
  requiring an incumbent remain undefined if one is missing; coverage and
  explicit bounds are reported instead of dropping those trajectories.
- Discovery intervals use 95% Wilson intervals. Supporting intervals use 2,000
  trajectory-level percentile bootstrap resamples, seed 0. These assume
  independent stochastic trajectories; seeded API requests cannot establish
  server-side independence. Repeated trajectories are retained and their first
  selections, entropy, duplicate-prefix and completed-trajectory rates reported.
- Models, prompts and reasoning/decoding protocols remain separate series.
  Analysis requires exact input directories and rejects legacy local-Chat or
  historical CSV protocols. RF-EI remains the principal adaptive reference;
  finite-candidate TPE, GP-EI and multimodal diversity remain comparators.
- Training costs are fixed-landscape L40S GPU-hours. Inference request latency,
  tokens and any server-reported allocation estimate are separate. No L40S/H200
  conversion or monetary inference cost is invented. Test accuracy/F1 are joined
  only offline for the validation-selected incumbent, with earliest-observed
  tie breaking.

## Smoke validation, production and storage

Run the tests in the existing environment; no installation is needed by this change:

```bash
cd /data/homes/55/jhedley/scGeneScope
poetry run pytest experiments/search/tests/test_llm_replay.py
poetry run pytest experiments/search/tests/test_search_baselines.py
```

The synthetic suite mocks inference/HTTP and uses temporary synthetic landscapes.
It includes unchanged Responses behavior, both original prompt variants,
standard-API-only operation, malformed outputs, retries, seed collisions,
immutable reuse, extension/reduction, corruption/orphans, concurrent writers,
atomic-write interruption, offline input preservation, failures and diversity.
These tests have been added/reviewed, **not executed**.

For a new execution protocol, validate smoke against the already running server.
For the existing compatible readable-directory smokes, this command reuses the
completed trajectories and prints their summary; no fresh inference is needed:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY  # Retain your actual key if the server requires one.
poetry run python experiments/search/replay_with_report.py \
  --llm-api chat --llm-model qwen3-8b-thinking \
  --methods llm_metrics llm_metrics_combined_gpu \
  --landscape full --budget 10 --seed 2000 \
  --thinking on --decoding-profile controlled \
  --temperature 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0 \
  --max-output-tokens 8192 --replicates 1 --smoke-test
```

Only after reviewing tests and both smoke certificates, the production command is:

```bash
poetry run python experiments/search/replay_with_report.py \
  --llm-api chat --llm-model qwen3-8b-thinking \
  --methods llm_metrics llm_metrics_combined_gpu \
  --landscape full --budget 10 --seed 2000 \
  --thinking on --decoding-profile controlled \
  --temperature 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0 \
  --max-output-tokens 8192 --replicates 50
```

This retains each compatible saved smoke and adds 49 trajectories per method, for 50 total.
Repeat the identical production command to resume. Keep the source tree, Python
environment, endpoint, model assertions and generation settings unchanged.

The exact result location, shared by the two method CSVs, is:

```text
experiments/search/results/lookup_replays/models/qwen3-8b-thinking/full/
  budget_010/seed_2000/thinking_on/
  controlled_t0.6_p0.95_k20_pp0_fp0_rp1_minp0_out8192/
```

It contains `metadata.json`, `response_identity.json`, `.writer.lock`, each
method's CSV and status CSV, `requests/`, `validation/`, `checkpoints/` and
`llm_logs/`. Errors are retained in journals, checkpoint interruption records
and terminal status exports. Failed evaluations are not filled with new samples.

The full metadata fingerprint remains enforced even when invisible settings
resolve to the same readable path. Errors list the changed fields and require
a separately named `--condition`, plus its own smoke. Nothing automatically
appends a hash. A condition name does not itself change the random seed schedule.

## Verify saving and resumption

1. After smoke, inspect both `validation/*.json` reports. Confirm completed
   `replicate_000.json` checkpoints, ten rows per method CSV, one completed row
   per status CSV, and request-ledger totals of one. Check recorded provenance
   limitations and requested parameters in metadata and journals.
2. Save checksums of the immutable smoke evidence:

   ```bash
   SCGS_QWEN8_RESULTS=experiments/search/results/lookup_replays/models/qwen3-8b-thinking/full/budget_010/seed_2000/thinking_on/controlled_t0.6_p0.95_k20_pp0_fp0_rp1_minp0_out8192
   sha256sum "$SCGS_QWEN8_RESULTS"/metadata.json \
     "$SCGS_QWEN8_RESULTS"/validation/*.json \
     "$SCGS_QWEN8_RESULTS"/checkpoints/*/replicate_000.json \
     "$SCGS_QWEN8_RESULTS"/llm_logs/*/replicate_000/*.json \
     > /tmp/scgs-qwen-smoke.sha256
   ```

3. Repeat the same smoke command. Expect explicit completed reuse for both
   methods and no new Chat Completions. Then run
   `sha256sum -c /tmp/scgs-qwen-smoke.sha256`. Model listing and optional
   provenance probes still occur; they do not generate tokens.
4. After the separately authorized production run, verify that smoke checksums
   still match. If all trajectories succeed, each method has IDs 0–49, 500 CSV
   observations and 50 completed status records. Failures produce fewer
   observations, retained failed status records and the same requested total 50.
5. For interruption, repeat the same command. Accepted selections remain saved;
   progress resumes after the saved prefix. Corruption or incompatibility stops
   explicitly. Preserve the files for inspection instead of deleting checkpoints
   or editing metadata to force reuse.
6. Run the exact offline comparison command in
   [LLM_EXPERIMENTS.md](LLM_EXPERIMENTS.md#offline-analysis). It opens classical
   inputs read-only. Reports go to
   `experiments/search/analysis/lookup_replay/llm_comparisons/inputs_<manifest-hash>/`.
   Confirm `requested_replicates=50`, coverage/failure counts and `is_final` for
   each LLM series; the classical series should each retain 1,000 trajectories.

## Older hash-directory smoke migration

No migration was performed. Copying the old hash-directory checkpoints into the new path
would **not** make them valid production replicates: the approved request seeds,
schema, checksums and source fingerprint changed. Do not edit their identity or
validation certificates to force compatibility. The successful readable-directory
smokes already use the current protocol and need neither migration nor fresh
inference for the reporting separation described above.

If an archival readable copy is later wanted, the separate approval procedure is:
stop writers; acquire a shared source lock; record a full source-file SHA256
manifest; copy the whole directory to a separately named **archive** destination;
compare every copied byte against that manifest; keep the source untouched and
mark the copy ineligible for this runner. This is archival copying only, not
scientific-protocol migration, and has not been authorized or executed here.

## Preservation and changed files

The initial working tree already contained uncommitted search work and changes
to `pyproject.toml` and `poetry.lock`. Those dependency files were not changed
by this audit. No staging or commits were performed.

Before/after SHA256 comparison verified **446 protected files unchanged**:
saved results and analysis, historical LLM outputs, both earlier smoke result
sets, landscapes, classical algorithms/storage/analysis/tests, and original
prompt/catalogue sources. The five named classical CSVs were inspected read-only
and each contains 1,000 distinct ten-evaluation replicates. Model-training code,
the original Responses clients and existing CLI defaults were not modified.

Files changed in this audit:

- `experiments/search/llm_client.py`
- `experiments/search/llm_replay_io.py`
- `experiments/search/llm_analysis.py`
- `experiments/search/replay_search.py` (restored to the exact readable-directory smoke source; local condition validation remains)
- `experiments/search/replay_with_report.py` (separate reporting launcher)
- `experiments/search/tests/test_llm_replay.py`
- `experiments/search/README.md`
- `experiments/search/METHODS.md`
- `experiments/search/LLM_EXPERIMENTS.md` (renamed and updated protocol guide)
- `experiments/search/LLM_AUDIT.md` (this report)

Planning-sequence terminology was removed from active local replay code, tests,
CLI output and documentation. Immutable historical output metadata and genuine
dataset acquisition identifiers in unrelated scripts were preserved.
