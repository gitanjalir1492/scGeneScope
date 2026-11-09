# LLM experiments: connect to an existing inference server

scGeneScope is an API client. It never starts, restarts, configures or stops vLLM.
Use the server you already run, with `OPENAI_BASE_URL`, `OPENAI_API_KEY` and
`--llm-model`. No development mode, scale-out flags, tokenizer-info endpoint,
specific vLLM version, local weight cache or shared server filesystem is required.

This is a new Chat Completions protocol on the unchanged 51-configuration
landscape, not an exact reproduction of the archived Responses experiments.
Historical Qwen/proprietary results and all completed classical sources, tests,
settings and results are preserved. No old trajectories are imported. Existing
CLI defaults remain unchanged, including 100 replicates and a Chat output limit
of 16,384 unless explicitly overridden.

Only read-only inspection and static code checks were performed during
implementation. Tests, inference and experiments below are for you to run.
The final review and saved smoke evidence are in [LLM_AUDIT.md](LLM_AUDIT.md).
The older hash-directory smokes used a different request-seed scheme and remain
isolated. The two successful smokes in the human-readable directory use the
current execution protocol and can count as the first production replicate.

Automatic reporting uses the separate `replay_with_report.py` launcher. It passes
the original arguments, environment and standard streams to `replay_search.py`
in a child process, then performs offline reporting only after successful exit.
The execution entry point is restored byte for byte to its saved smoke hash;
all other recorded execution sources also match. No new smoke, condition name,
directory, metadata edit or compatibility exception is required for reporting.
Normal checks still refuse any changed generation settings, source or verifiable
server provenance when you run the command.

Keep reporting changes in the launcher and offline analysis modules. Full hashes
of these sources are recorded separately in `comparison_metadata.json`; they
change report provenance, not experimental compatibility. The complete
`replay_search.py` and all other execution sources remain fingerprinted. The
original entry point can still be run directly, followed by `--analyze-only`.

## Declared comparisons

The primary outcome is **optimum discovery within ten evaluations**. Mean best
validation accuracy is the key supporting outcome. RF-EI is the principal
adaptive comparator; finite-candidate TPE, GP-EI and multimodal diversity are additional
prespecified comparators. Report top-three discovery
(including ties at third place), simple regret, training GPU-hours, separate
inference usage/time, failures, and offline test accuracy/F1 of the
validation-selected incumbent.

| Model / condition | Methods | Total trajectories per method | Budget |
| --- | --- | ---: | ---: |
| Qwen3-8B thinking | `llm_metrics`, `llm_metrics_combined_gpu` | 50 fresh | 10 |
| Qwen3.8-27B thinking, xhigh | both metrics methods | 10 | 10 |
| Qwen3.8-27B non-thinking, controlled | `llm_metrics` | 10 | 10 |
| Qwen3.8-27B non-thinking, publisher decoding (secondary) | `llm_metrics` | 10 | 10 |
| GPT-OSS-20B medium | both metrics methods | 10 | 10 |
| GPT-OSS-20B low, high | `llm_metrics` | 10 each | 10 |
| Later: GPT-OSS-120B medium; Gemma-4-31B-IT thinking | both metrics methods | 10 | 10 |
| Later: GPT-OSS-120B low/high; Gemma thinking off | `llm_metrics` | 10 each | 10 |

Each model/prompt/reasoning/decoding condition needs a successful **one-trajectory,
ten-evaluation smoke run**. It is replicate 0 and counts toward the declared
total: production at 50 adds 49. The gate uses technical validity only, never
search accuracy. Ten trajectories remain exploratory rather than a precise ranking.

## 1. Run the synthetic tests

In the existing repository Poetry environment:

```bash
cd /data/homes/55/jhedley/scGeneScope
poetry run pytest experiments/search/tests/test_llm_replay.py
poetry run pytest experiments/search/tests/test_search_baselines.py
```

All HTTP and inference in the new tests are mocked. Tests cover a server exposing
only standard model-listing and Chat endpoints, missing optional metadata,
endpoint/configuration separation, provenance assertions, original Responses
behavior, fixed limits, invalid/duplicate outputs, smoke gates, partial resume,
received-response recovery, archive preservation and offline analysis.
No actual server, model download or measured-landscape search is used.

## 2. Connect to your current Qwen3-8B server

Leave the server running. In the replay shell, set the URL of its API and the
key it expects. For your existing loopback endpoint:

```bash
cd /data/homes/55/jhedley/scGeneScope
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

Remote HTTP(S) OpenAI-compatible URLs are also supported. Credentials belong in
`OPENAI_API_KEY`, not in the URL; credentials are never stored in metadata.
The normalized API URL is part of the protocol identity.

Choose the output cap explicitly before smoke. **A server context of 16,384
cannot fit 16,384 output tokens plus this prompt.** The example below uses an
explicit 8,192-token starting cap for your existing 16,384-context server.
This is a documented new protocol setting, not an automatic change to the
default and not an assertion that 8,192 is sufficient for every response.
The smoke must establish that final answers fit.

```bash
SCGS_REPLAY_ARGS=(
  --llm-api chat --landscape full --budget 10 --seed 2000
  --max-output-tokens 8192
)
SCGS_QWEN8_ARGS=(
  --llm-model qwen3-8b-thinking --thinking on
  --decoding-profile controlled
  --temperature 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0
)

poetry run python experiments/search/replay_with_report.py \
  "${SCGS_REPLAY_ARGS[@]}" "${SCGS_QWEN8_ARGS[@]}" \
  --methods llm_metrics llm_metrics_combined_gpu --replicates 1 --smoke-test
```

For the two already successful readable-directory smokes, this repeats validation
and prints the summary while reusing the completed trajectories. Reporting does
not require another smoke. The production command below can reuse those smokes
directly if all execution settings and verifiable server provenance still match.

If your server has a larger context and you choose the original 16,384 cap,
change the array explicitly **before** smoke. No parameter is silently reduced,
removed or replaced when an endpoint rejects it. A changed cap or other setting
creates a new protocol and requires another smoke.

The adapter checks the served alias through `GET /v1/models`. The smoke uses the
original prompts in real Chat requests and verifies valid final JSON, unique
configuration IDs, finish reasons, measured token usage, reasoning channels,
request seeds and checkpoint reloads. Chat completion acceptance is tested by
these journalled requests, not by an unlogged extra inference call.

## 3. Run and safely resume the fresh 50-trajectory conditions

After both smoke validations succeed:

```bash
poetry run python experiments/search/replay_with_report.py \
  "${SCGS_REPLAY_ARGS[@]}" "${SCGS_QWEN8_ARGS[@]}" \
  --methods llm_metrics llm_metrics_combined_gpu --replicates 50
```

**Repeat the identical production command to resume.** Completed trajectories
are skipped; partial trajectories continue at the next evaluation. A response
already received and journalled before interruption is reused. Smaller requested
counts never delete saved observations. Resume an interrupted smoke by repeating
the smoke command, not by bypassing it.
Reused checkpoints explicitly print `reused (completed; no inference sent)`.
A completed or terminally failed replicate is never replaced by rerunning a
command. Smoke and production use the same identity; only the requested total
and smoke-validation gate differ.

On normal completion, including a run that reuses completed checkpoints, the
launcher automatically runs offline analysis after the execution child exits
and releases the replay writer lock.
It prints a compact budget-endpoint table with model/prompt, evaluation number,
requested replicate count, mean best validation accuracy and 95% intervals,
simple regret, optimum/top-three discovery and intervals, cumulative training
GPU-hours, and validation-selected incumbent test accuracy/F1. A separate table
shows inference tokens/time, request and trajectory failures, duplicate complete
trajectories and distinct first selections. Settings and exact input directories
identify each table row; compatible methods share a directory but remain separate.

The automatic report reads only that run's explicit directory and selected
methods. It includes the full saved requested count even if a later invocation
asks for fewer replicates. It does not automatically load historical runs or
classical comparisons. A reporting error cannot undo already saved trajectories;
use `--analyze-only` below to retry reporting without inference. Interrupted
inference can also be inspected offline; final estimates remain withheld for
unfinished conditions.

Qwen3-8B's omitted sampling arguments resolve to temperature 0.6, top-p 0.95,
top-k 20 and presence penalty 0.0. The fixed additional values are frequency
penalty 0.0, repetition penalty 1.0 and min-p 0.0. These are sent explicitly and
included in protocol metadata, so explicitly supplying those same defaults
does not define a new condition. The Chat output default remains 16,384; these
commands explicitly select 8,192. Thinking is requested through
`chat_template_kwargs.enable_thinking`, with no added scientific instructions
or `/think` text. This is the interface documented by
[Qwen](https://qwen.readthedocs.io/en/stable/deployment/vllm.html).

## 4. Additional models and reasoning conditions

Manage servers however you normally do. Once the required alias is served,
set `OPENAI_BASE_URL` to that endpoint and use the following client commands.
The examples retain the output cap chosen above. Within the controlled Qwen
on/off comparison, keep the cap and all sampling parameters identical. Document
any cap difference across models as a protocol difference.

For Qwen3.8-27B thinking, after it is available through your endpoint:

```bash
SCGS_QWEN38_ON=(
  --llm-model qwen3.8-27b --thinking on --reasoning-effort xhigh
  --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0
)
poetry run python experiments/search/replay_with_report.py \
  "${SCGS_REPLAY_ARGS[@]}" "${SCGS_QWEN38_ON[@]}" \
  --methods llm_metrics --replicates 1 --smoke-test
poetry run python experiments/search/replay_with_report.py \
  "${SCGS_REPLAY_ARGS[@]}" "${SCGS_QWEN38_ON[@]}" \
  --methods llm_metrics --replicates 10

poetry run python experiments/search/replay_with_report.py \
  "${SCGS_REPLAY_ARGS[@]}" "${SCGS_QWEN38_ON[@]}" \
  --methods llm_metrics_combined_gpu --replicates 1 --smoke-test
poetry run python experiments/search/replay_with_report.py \
  "${SCGS_REPLAY_ARGS[@]}" "${SCGS_QWEN38_ON[@]}" \
  --methods llm_metrics_combined_gpu --replicates 10
```

The thinking-on runs above also form the on arm. The off arms are:

```bash
for SCGS_PROFILE in controlled publisher-nonthinking; do
  poetry run python experiments/search/replay_with_report.py \
    "${SCGS_REPLAY_ARGS[@]}" --llm-model qwen3.8-27b --thinking off \
    --decoding-profile "$SCGS_PROFILE" \
    --methods llm_metrics --replicates 1 --smoke-test || break
  poetry run python experiments/search/replay_with_report.py \
    "${SCGS_REPLAY_ARGS[@]}" --llm-model qwen3.8-27b --thinking off \
    --decoding-profile "$SCGS_PROFILE" \
    --methods llm_metrics --replicates 10 || break
done
```

The controlled profile uses temperature 1.0, top-p 0.95, top-k 20 and presence
penalty 0.0 in both arms. The separately labelled publisher profile uses
0.7 / 0.8 / 20 / 1.5 and rejects conflicting overrides. It does not isolate
reasoning alone. Qwen3.8 effort accepts `low`, `medium` and `xhigh`, not `high`.

For a served GPT-OSS-20B endpoint:

```bash
for SCGS_EFFORT in medium low high; do
  poetry run python experiments/search/replay_with_report.py \
    "${SCGS_REPLAY_ARGS[@]}" --llm-model gpt-oss-20b \
    --reasoning-effort "$SCGS_EFFORT" \
    --methods llm_metrics --replicates 1 --smoke-test || break
  poetry run python experiments/search/replay_with_report.py \
    "${SCGS_REPLAY_ARGS[@]}" --llm-model gpt-oss-20b \
    --reasoning-effort "$SCGS_EFFORT" \
    --methods llm_metrics --replicates 10 || break
done

poetry run python experiments/search/replay_with_report.py \
  "${SCGS_REPLAY_ARGS[@]}" --llm-model gpt-oss-20b --reasoning-effort medium \
  --methods llm_metrics_combined_gpu --replicates 1 --smoke-test
poetry run python experiments/search/replay_with_report.py \
  "${SCGS_REPLAY_ARGS[@]}" --llm-model gpt-oss-20b --reasoning-effort medium \
  --methods llm_metrics_combined_gpu --replicates 10
```

GPT-OSS uses 1.0 / 1.0 / unrestricted top-k / 0.0 across efforts. Effort is sent
as `reasoning_effort`, not inferred from maximum output tokens or response
length. Its native server-added Harmony preamble/date may affect reproducibility;
when unavailable, that limitation is explicitly recorded.

Prepared aliases are `gpt-oss-120b` (same effort controls),
`gemma-4-31b-it` (thinking on/off; defaults 1.0 / 0.95 / 64 / 0.0), and optional
`glm-4.7-flash` (thinking on/off). The corresponding published identifiers are
`openai/gpt-oss-120b`, `google/gemma-4-31B-it` and `zai-org/GLM-4.7-Flash`.
Known published identifiers can themselves be passed as `--llm-model` if that is
the ID returned by your API. These are compatibility profiles, not evidence
that the named checkpoint is loaded. Each condition must pass its own smoke.

## Compatibility and provenance: what is actually verified

Only model listing and Chat Completions are mandatory. Optional queries to
`/version`, `/server_info`, `/tokenizer_info`, `/openapi.json` and the Chat
render endpoint collect evidence if accessible. They do not generate tokens,
modify the server or require flags to be enabled. Missing, inaccessible or
unrecognized diagnostic responses are recorded as unavailable.

Every provenance field has a value, source and status:

- **verified**: directly observed or reported by the endpoint. For example,
  model-list membership, an advertised context limit or a rendered sampling
  value. This is not an independent audit of model weights or GPU execution.
- **user_supplied**: explicitly asserted by the caller, such as optional
  `--model-revision <40-character-commit>` or `--context-window 16384`.
  A conflicting server report stops the run. These flags do not configure vLLM.
- **unavailable**: no trustworthy value was obtained. No model revision,
  tokenizer, server version, hardware or inference setting is fabricated.

An immutable model revision is no longer guessed from a cached checkpoint or
hard-coded release. A server-reported mutable branch such as `main` is retained
as a branch report, not promoted to a verified immutable revision.

The smoke separately records:

- exact requested parameters and seeds;
- successful API acceptance and output parsing;
- observed reasoning/final channels and final-answer validity;
- effective sampling/render evidence, if the server supplies it;
- known context headroom, or the explicit limitation if context is unknown.

**API acceptance is not proof that every parameter was applied internally.**
Schema-declared fields are reported separately from effective settings. Explicit
schema rejection, API rejection, contradictory rendered settings, wrong model,
unexpected reasoning in an off condition, truncated output or invalid selections
cannot silently fall back to a different configuration. The implementation never
strips a rejected parameter or parses a reasoning trace as the final answer.

Without rendering evidence, seed application and effort application may remain
unverified even when requests succeed. Those limitations are printed as
`PROVENANCE:` messages, saved in metadata/smoke reports and exposed in the
analysis tables. Do not describe such a run as engine-verified or infer that an
effort setting worked merely because its responses are longer. A server with
invariant aliases but unreported weight revisions cannot provide proof that its
backend stayed identical across restarts; operator records remain necessary for
that stronger claim.

When a context limit is known, the fixed output cap plus actual prompt usage
and 256 tokens of margin must fit. If rendered prompt counts are unavailable,
the smoke checks the returned prompt-token usage instead. An unknown context
limit is flagged; it is never replaced by an assumed 32,768. Missing token usage
or failure to complete a final answer blocks smoke. You can explicitly assert a
known context through `--context-window`; it remains labelled user-supplied
unless the API confirms it.

## Protocol consistency, storage and failures

The runner derives its directory from resolved settings. For the commands above:

```text
experiments/search/results/lookup_replays/models/qwen3-8b-thinking/full/
  budget_010/seed_2000/thinking_on/
    controlled_t0.6_p0.95_k20_pp0_fp0_rp1_minp0_out8192/
      metadata.json
      .writer.lock
      response_identity.json
      <method>.csv
      <method>_status.csv
      requests/<method>.json
      validation/<method>.json
      checkpoints/<method>/replicate_000.json
      llm_logs/<method>/replicate_000/trial_001_decision_attempt_01_transport_0000.json
```

`pp`, `fp`, `rp` and `minp` mean presence penalty, frequency penalty, repetition
penalty and minimum probability. Effort-controlled models add `_effort_<value>`
to the thinking component. Published model identifiers encode `/` as `%2F` in
one directory component. `--condition descriptive-name` adds a final
`condition_descriptive-name/` component for separately declared conditions.
This name is a label, not a source of independent seeds; use a different base
seed for a separately sampled replication.

Identity includes the API endpoint, served alias/profile, explicit revision/context
assertions, reasoning, sampling, cap, seed, budget, landscape/source/prompt hashes,
client Python version and recorded server evidence. Replicate count and method
subset do not change identity; API credentials are excluded. The full SHA256
fingerprint stays in `metadata.json`, never in the LLM result directory name.
Changing visible settings selects another directory. Changes to an endpoint,
revision assertion, source code or other metadata can resolve to the same path:
the writer then lists the differing fields and refuses before any generation.
Choose an explicit, separately named `--condition` and run a fresh smoke.
It never appends a hash, overwrites incompatible files or automatically migrates
old results. The classical directory structure is unchanged.

This conservative identity also separates changed optional provenance. If a
previously available diagnostic disappears or new evidence becomes available,
the changed fingerprint requires a fresh smoke rather than silently combining
different evidence levels. The runner prints the directory and provenance.
Previously verified effective sampling must remain available during that run;
otherwise it stops with the prefix intact. Do not edit/copy metadata to force a
resume. Servers with no diagnostic endpoints can run normally with consistent,
explicitly unavailable provenance.

Checkpoints are checksummed, atomic and authoritative; CSVs export all saved prefixes
atomically. File contents and parent directory entries are flushed before proceeding.
A writer lock prevents concurrent writers to one condition. Requests
are journalled before submission; raw responses are saved immediately on receipt.
Exact messages, seeds, sampling settings, available rendering evidence, latency,
usage and failures are retained. The original prompts, candidate shuffling and
observed GPU-cost feedback remain unchanged.
Before generation, all saved checkpoints, requests and exports are checked,
including saved replicate IDs outside a reduced requested count. Corrupt JSON,
orphaned logs, missing ledgers, changed smoke evidence and mismatched observations
stop with files preserved. Uncommitted `.tmp` files are never used as observations.
Received responses are matched to exact request payloads; accepted rows are
matched to their decision journals and reconstructed observed-history prompts.
If the standard API reports a `system_fingerprint`, its value is frozen in
`response_identity.json`; a changed or disappearing value stops the run. This
is a server report, not verification of weights.

Missing final text, non-stop finish reasons, malformed JSON, invalid IDs or
duplicates get at most three logged attempts using distinct declared seeds and
the same prompt. They consume inference cost but no architecture evaluations.
Exhaustion marks a terminal failure; it is not replaced by a fresh replicate.
Smoke refuses even recovered malformed attempts, so failures are inspected before
production. Do not discard a technically valid trajectory because of its accuracy.

Transport retries reuse the same payload and seed, up to three per invocation
for transient failures. A non-retryable HTTP error stops with the response/error
logged; no fields are dropped. Manual resumption can repeat an interrupted
in-flight call, whose original usage remains unknown. Exact-once inference and
bitwise reproducibility are not claimed. Exhausted transport retries leave an
explicitly interrupted `in_progress` trajectory, with logged errors; resume it
with the same command. Final discovery estimates remain withheld until it is
resolved. Exhausted invalid-output attempts are terminal `failed` trajectories.

Request seeds use the first eight hex digits of SHA256 over compact JSON
`["llm-replay-request-v1", base_seed, method, replicate, trial, attempt]`,
interpreted as an unsigned 32-bit integer. Summary requests use `method + ":summary"`.
Replicates are zero-indexed, trials and output attempts are one-indexed. The
catalogue shuffle seed is `base_seed + replicate`. Both seeds are recorded.
Before any generation, every potential request seed in the requested and already
saved method/count schedule is checked for collisions. A collision stops the
command without altering the seed schedule or replacing a trajectory; choose a
different base seed and validate a new smoke. Extension rechecks the full schedule.
Model/effort are omitted to give controlled arms the same seed schedule; that
does not establish paired stochastic outcomes across different model families.
Keep repeated trajectories in the analysis and report their concentration.
Independent-trajectory intervals assume independently seeded stochastic sampling;
standard-API acceptance alone cannot verify that the server applies seeds or
establish stochastic independence. Journal replay is recovery, not another sample.

## Offline analysis

To regenerate the automatic 50-replicate summary without any server connection:

```bash
SCGS_QWEN8_RESULTS=experiments/search/results/lookup_replays/models/qwen3-8b-thinking/full/budget_010/seed_2000/thinking_on/controlled_t0.6_p0.95_k20_pp0_fp0_rp1_minp0_out8192
poetry run python experiments/search/replay_search.py --analyze-only \
  --methods llm_metrics llm_metrics_combined_gpu \
  --landscape full --budget 10 --replicates 50 \
  --analysis-inputs "$SCGS_QWEN8_RESULTS"
```

The same directory contains the already completed smokes and their production
extensions; no reporting-specific condition name is used.
The offline replicate limit is a ceiling, not a request to create trajectories.
The displayed evaluation number is the declared budget endpoint; all-evaluation
curves are saved separately. Wilson intervals for discovery and the existing
trajectory bootstrap for supporting means are unchanged. A single smoke has
no bootstrap confidence interval; missing or unfinished estimates print `NA`.

The consolidated report reads the existing classical CSVs directly; it does not
rerun the optimisers or require newly generated classical analysis files. The
saved budget-ten baseline directory below contains all ten classical methods
with 1,000 completed replicates each. On 2026-11-04, the Qwen directory contains
50 completed replicates for each of the two prompting conditions.

The existing `openai_gpt6_astra` and `anthropic_claude_opus_5_5` archives each
contain seven prompting conditions with ten completed trajectories per condition.
They can also be selected explicitly for the consolidated table. Their missing
execution provenance and failure/request accounting remain labelled unavailable;
they are never adopted into the new local replay protocol.

Stop writing the input conditions before analysis. To include all ten classical
methods, both recent Qwen prompts and all saved Astra/Opus prompts, run from the
repository root:

```bash
SCGS_CLASSICAL=experiments/search/results/lookup_replays/baselines/full/budget_010_startup_003_seed_1000_pa8604c3c37c3
SCGS_QWEN8_RESULTS=experiments/search/results/lookup_replays/models/qwen3-8b-thinking/full/budget_010/seed_2000/thinking_on/controlled_t0.6_p0.95_k20_pp0_fp0_rp1_minp0_out8192
SCGS_QWEN27_XHIGH=experiments/search/results/lookup_replays/models/qwen3.8-27b/full/budget_010/seed_2000/thinking_on_effort_xhigh/controlled_t0.6_p0.95_k20_pp0_fp0_rp1_minp0_out16384
SCGS_ASTRA=experiments/search/results/lookup_replays/models/openai_gpt6_astra
SCGS_OPUS=experiments/search/results/lookup_replays/models/anthropic_claude_opus_5_5

poetry run python experiments/search/replay_search.py --analyze-only \
  --methods rf_ei tpe_discrete gp_ei multimodal_diverse \
    random stratified_random multimodal_random bayesian \
    tpe_repaired_hierarchical tpe_repaired_uniform \
    llm_metrics llm_metrics_combined_gpu llm_summary llm_metrics_shuffled \
    llm_metrics_budget llm_metrics_no_isolation llm_metrics_combined \
  --landscape full --budget 10 --replicates 1000 \
  --analysis-inputs "$SCGS_CLASSICAL" "$SCGS_QWEN8_RESULTS" "$SCGS_ASTRA" "$SCGS_OPUS" \
    "$SCGS_QWEN27_XHIGH::llm_metrics"
```

This produces 32 separate summary rows: ten classical rows at N=1,000, seven
Qwen3-8B rows at N=50, seven Astra rows at N=10, seven Opus rows at N=10,
and the Qwen3.8-27B xhigh `llm_metrics` row at N=50.
RF-EI remains the principal adaptive reference. No baseline
rerun, LLM server or API credentials are needed. Input CSVs, checkpoints, journals
and metadata are opened read-only; only derived analysis files are written.
The `PATH::method,method` analysis-input suffix is a reporting-only filter. It is
used above because the 27B directory also contains smoke/partial state for the
other six methods; those rows do not enter this table. Methods after `::` must
also appear in the global `--methods` list. Input selection and exact selected
methods are saved in comparison metadata. No execution metadata is bypassed.

`--replicates 1000` is a ceiling on zero-based replicate IDs for **every input**,
not a request to create trajectories or to expand the Qwen sample. Setting it to
50 would restrict the classical rows to their first 50 replicates too. The loader
now prints the saved and included population for each series and explicitly
reports any exclusions. `input_coverage.csv` and the analysis manifest record
these counts. For local Chat runs, the saved population is the request-ledger count, including
unfinished/not-started replicates; for classical CSVs it is the completed count.
For historical Astra/Opus CSVs it is the saved completed count, not an inferred
original request count.

Append other exact condition directories when available. Full protocol hashes
keep models, prompts, reasoning and sampling settings separate; copied instances
of the same protocol/method are rejected. A requested method missing from all
inputs is an error, so a missing comparator cannot silently disappear. Archived
Qwen runs lack verified provenance and are not adopted into the new protocol;
the historical analysis path remains available. The automatic post-run report
still covers just the current LLM condition; this explicit offline command adds
the saved classical and explicitly selected historical comparisons.

Historical CSV rows use the same full-landscape scoring, validation-selected
test outcomes, Wilson discovery intervals and trajectory bootstrap as the other
rows. Their statistics describe the saved completed sample only: no original
failure/request ledger exists, so failed/not-started counts, failure rates, tokens,
latency and costs are `NA`, not zero. Text-only historical logs are included in
the input hashes but do not establish complete request accounting. Selection
diversity remains measurable from the saved trajectories. Sampling/reasoning
settings, revision and execution fingerprint are labelled unrecorded. Each
historical source gets an analysis snapshot hash; its `protocol_hash` is empty.
The snapshot identifies archive contents, not a verified scientific protocol.

This adapter is restricted to the two explicitly selected historical model
directories above. It validates every CSV row and rejects incomplete trajectories,
duplicates, wrong landscapes/budgets and feedback mismatches. No source metadata
or lock files are created. Since historical writers did not use locks, the loader
scores an in-memory snapshot and checks for file changes before saving a report;
stop any writer first. Metadata-bearing protocols retain their existing strict
fingerprint, lock and checkpoint checks. Missing modern metadata never triggers
historical fallback. No historical Qwen results are automatically included.

Reports under `analysis/lookup_replay/llm_comparisons/inputs_<hash>/` include:

- `input_coverage.csv`: population source, saved/included counts, exclusions and
  replicate-ID ceiling for each full protocol/method or historical snapshot identity.
- `final_summary.csv`: numeric core columns from the automatically printed
  endpoint table, separate confidence-bound columns, full protocol identity and
  completion/failure coverage. These are the existing primary-outcome estimates.
- `inference_diagnostics.csv`: per-trajectory token/latency means, request/failure
  counts and rates, missing telemetry, completed-trajectory duplication and
  first-selection frequencies/entropy. Duplicate rates are labelled conditional
  on completion; inference time is separate from training GPU-hours.
- `primary_outcomes.csv`: discovery with Wilson intervals, supporting means and
  bootstrap intervals, failure/coverage counts, separate inference costs and
  provenance status columns. All final means, primary rates and confidence
  intervals are withheld while requested trajectories are unfinished. Terminal
  failures remain in the denominator, including hits achieved before failure.
  A failure with no incumbent makes the unconditional accuracy mean undefined;
  report coverage and the explicit lower/upper bounds, not a successful-only mean.
- `trajectory_endpoints.csv`: every requested trajectory, including not-started,
  partial and failed ones, and offline test performance of the validation-selected incumbent.
- `all_trajectory_curves.csv`, `all_trajectory_compute.csv`: primary evaluation
  and common-training-compute curves including failures. Terminal failures retain
  their last accepted incumbent and cost; zero-evaluation failures retain no
  incumbent. Unfinished conditions have no final curves or intervals. No mean is
  computed over a favourable subset with available incumbents.
- `trajectory_diversity.csv`: duplicate ordered trajectories/sets and first-choice
  frequencies/entropy. Repeated samples are not deduplicated.
- `comparison_metadata.json`: settings, provenance and exact input/source hashes,
  including full reporting-source SHA256 values separately from the unchanged
  execution source hashes preserved in each input's metadata.
- `completed_trajectory_curves/`: existing accuracy/regret/hit/compute plots and
  pointwise trajectory-bootstrap intervals, explicitly conditional on completion.

Inference costs use measured token counts and request latency. Request-hours are
wall time / 3600. GPU-hours are estimated only when server GPU parallelism is
reported; otherwise they are unknown, not assumed to be one H200. Even with a
reported GPU count, this is an allocation estimate rather than measured GPU
utilisation or money. Loading, idle allocation, render calls and unknown in-flight
work are excluded. Monetary cost is unavailable without billing or price data.
Classical surrogate CPU time was not measured for the completed classical runs.
Keep inference allocation on H200 (operator-reported; verify separately) distinct
from looked-up scGeneScope training GPU-hours on one L40S. No hardware equivalence
or conversion is assumed.

Test outcomes are joined only offline after choosing the incumbent by observed
validation accuracy, with earliest-observed tie breaking. No new dependencies,
server manager, weight downloads, training changes or later prompt-factorial
experiments are introduced.
