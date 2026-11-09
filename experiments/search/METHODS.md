# Methods

## Local LLM protocol

[LLM_EXPERIMENTS.md](LLM_EXPERIMENTS.md) declares the new experiment matrix, generation settings,
reasoning controls, smoke criteria, retry/failure policy, seeds and analysis.
The client connects to an existing OpenAI-compatible endpoint without managing
the server. Diagnostic metadata is optional; requested controls, API acceptance
and independently observed effective settings are recorded separately. Unknown
revision, reasoning-effort or sampling application is not claimed as verified.
The primary comparison is optimum discovery within ten evaluations; mean best
validation accuracy is the key supporting outcome. RF-EI is the principal
adaptive classical comparator, alongside finite TPE, GP-EI and multimodal diversity.
Top-three discovery, training GPU-hours, separate inference costs, failure rates
and offline validation-selected test accuracy/F1 are additional outcomes.

The Qwen3-8B replications are 50 fresh trajectories per prompt, not extensions of
the archived ten. Local Chat uses the existing user prompts verbatim but a new
explicitly identified API/serving/reasoning/decoding protocol. The primary
Qwen3.8 thinking switch uses matched sampling; publisher-recommended non-thinking
decoding is a separate secondary condition. Technical smoke validation includes
the complete ten-evaluation trajectory and never gates on search performance.
Failed trajectories are retained in the primary discovery denominator. Test
results are accessible only to offline analysis. The classical protocol below
and the archived Responses clients are unchanged.

Local results use readable model/landscape/budget/seed/reasoning/decoding paths.
Full protocol fingerprints remain enforced through metadata: incompatible
settings at the same path produce a field-level error before generation, and
require a separately named condition plus fresh smoke. The revised seed scheme
is documented in [LLM_EXPERIMENTS.md](LLM_EXPERIMENTS.md); older hash-directory
smokes remain isolated. The successful readable-directory smokes already use
the current execution protocol and can extend to production. The optional
`replay_with_report.py` launcher runs that unchanged, fully fingerprinted
execution entry point before offline reporting; reporting-source hashes are
audited separately and do not define a scientific condition. Completed checkpoints are reused
explicitly and never regenerated. Final estimates and intervals are withheld
until every requested trajectory is completed or terminally failed; failures
remain in primary discovery and compute-curve denominators. See
[LLM_AUDIT.md](LLM_AUDIT.md) for the final audit and handover.

## Extended full-landscape classical comparison

This section defines the new classical replay protocol. Later sections document
the historical pilot and LLM experiments; their references to future training
do not apply to this comparison. No further training or datasets are required.

All feedback comes from the fixed 51-configuration landscape. A selected
configuration consumes one evaluation; duplicates are prohibited. Proposers
receive architectural descriptions and completed validation feedback only.
New selectors cannot read result files and are not passed the evaluator or any
test scores, global ranks, unobserved scores or measured unobserved costs.
Feedback retains the existing six-decimal validation precision.

### Random and structural policies

`random` preserves the original sampler and its seeded budget-sized sample.
`stratified_random` shuffles each family's configurations and selects one from
each nonempty family in a newly shuffled family order each round. At budget ten
this gives a random 4/3/3 allocation. Exhausted families are skipped.

`multimodal_random` uses a seeded uniform permutation of the 36 multimodal
configurations. `multimodal_diverse` chooses its first configuration uniformly,
then maximises minimum categorical distance to the selected set, with seeded
tie-breaking. Its four equally weighted decisions are RNA encoder, image
encoder, profile/aggregation and fusion. It never uses observed score values.
Neither method preferentially ranks component values or uses trial-position
rules. Both deliberately restrict eligibility and reject budgets above 36.

For uniform sampling of b configurations from N with K targets, the exact hit
probability is `1 - comb(N-K, b)/comb(N, b)`. At b=10, unrestricted random finds
the unique optimum with probability 10/51 and a top-three configuration with
probability 0.488115. Multimodal random uses N=36 and only targets inside its
subset; an optimum outside that subset has zero discovery probability.

### Original and repaired hierarchical TPE

`bayesian` is unchanged. Initially it weights the three families equally and
single/multi-profile equally, so it is not uniform over the 51 configurations.
It records rejected duplicates as PRUNED; Optuna includes these in startup
accounting and density fitting. Its historical study reconstruction and fixed
reseeding are preserved. It can still fail after its original retry limit.

The repaired variants reuse the exact conditional parameterisation, independent
Optuna TPE, sampler defaults, and reconstruction/reseeding convention. Rejected
proposals are closed as FAIL, excluded from TPE's observations. Only genuine
completed evaluations count towards startup. `tpe_repaired_hierarchical` keeps
the original hierarchical random prior. `tpe_repaired_uniform` supplies uniform
configuration startup, recorded in the original hierarchical parameterisation.

After 255 rejected proposals the repaired variants draw uniformly from the
remaining configurations. This produces one evaluation, not 255, and the actual
feedback is available to the next selection. `proposal_attempts` and `fallback`
are logged. Rejection no longer penalises duplicates in the density model, so
fallback can be frequent and its rate is part of the interpretation.

### Finite TPE and structural encoding

New surrogates use five categorical blocks: family, RNA encoder, image encoder,
profile/aggregation, fusion. Inactive encoders/fusion have a separate category.
Profile/aggregation has three choices (single, multi-average, multi-Transformer).
This avoids double-counting the profile decision. Numbered IDs are never model
features. The one-hot form has 16 columns, scaled by 1/sqrt(2), so one block
mismatch contributes one squared-distance unit. Fitting uses observed targets
only; constructing structural features for all candidates does not expose scores.

`tpe_discrete` fits multivariate categorical mixture densities using Optuna 4.9.0
and evaluates `log l(x) - log g(x)` over every remaining valid configuration.
Good-group size and chronological weights follow that version's default TPE;
at the main budget this means one elite observation. The prior and categorical
smoothing are Optuna's. Selection is an exhaustive maximum with seeded ties.
The density estimator's private API is isolated in `tpe_acquisition.py`, which
rejects other Optuna versions. A flat opaque categorical ID would leave all unseen
IDs exchangeable and is intentionally not used. This method changes representation,
joint modelling and acquisition optimisation, not just startup distribution.

### GP and RF acquisition

`gp_ei` fits ConstantKernel times Matérn (nu=2.5), with one shared length scale.
Initial amplitude/length are 1; bounds are [1e-3, 1e3] and [0.1, 10]. Targets are
normalised from observed data only. Three seeded optimisation restarts are used.
Jitter starts at 1e-8, then 1e-6 and 1e-4 on numerical failure; the used value is
recorded. Failure at all levels raises an error rather than silently switching
to random search. A large ARD parameterisation is avoided with such little data.

`rf_ei` retrains 256 bootstrap regression trees each step, with max_features=1.0,
min_samples_leaf=1, otherwise standard sklearn defaults, and one fitting thread.
Predictive mean and sample standard deviation (ddof=1) are computed over tree
predictions. This deviation is not divided by sqrt(number of trees). Gaussian
moment-matched EI is an approximation, not a calibrated Bayesian posterior or
the complete SMAC algorithm. Averaging individual-tree positive improvements
would be degenerate because tree predictions cannot exceed the observed maximum.

Both use maximisation EI with zero exploration offset:
`EI = (mu-best)*Phi(z) + sigma*phi(z)`, `z=(mu-best)/sigma`.
Zero variance uses the exact limit `max(mu-best, 0)`. Only unobserved valid
candidates are scored. If every EI underflows to zero, maximum predictive
standard deviation is used, followed by seeded ties. When only one eligible
candidate remains, it is selected directly without unnecessary model fitting.

### Initialisation, seeds and scope

Primary comparisons use natural starts, with three startup evaluations for the
new Bayesian methods, counted within budget. Uniform-start methods share a
seeded catalogue-permutation prefix. Startup counts 1 and 5 are prespecified
sensitivities; counts at or above budget leave no adaptive phase. Original TPE's
nominal startup count does not guarantee that many distinct random evaluations.
Matched-start comparisons are a separate protocol; LLM integration is deferred.

Replicate r uses base_seed+r (default base 1000). Proposal tie-breaking uses local
seed streams; GP/RF fitting seeds depend on base replicate seed and observation
count. Method order cannot affect selections. Python and numerical package
versions are part of the saved protocol identity. Existing LLM API sampling is
not controlled by the local seed; prompts/responses remain its audit record.
The CLI still defaults to 100 replicates. The planned classical comparison uses
explicit `--replicates 1000`.

### Saving and offline analysis

New classical results use settings-derived directories below
`results/lookup_replays/baselines/`, with one CSV per method and a compact
metadata sidecar. Selection code, landscape, settings and dependency identities
prevent incompatible data from sharing a directory. Replicate count is not an
identity field. Existing complete replicates are checked and reused, missing
IDs are appended atomically, and lowering the request never truncates files.
Malformed partial CSV trajectories fail closed; an interrupted in-memory
replicate can be rerun. Historical root/model results are never adopted as new
baseline output files. See README for exact commands and paths.

Analysis reports best full-validation accuracy, simple regret, tie-inclusive
optimum/top-three hits and cumulative recorded single-GPU training time. Accuracy,
regret, cost and validation-selected-incumbent test metrics receive pointwise
95% percentile bootstrap intervals (2,000 whole-trajectory resamples). Discovery
probabilities receive Wilson intervals, including nonzero uncertainty at 0/N
and N/N. These are pointwise intervals, not simultaneous confidence bands.

Compute curves evaluate each trajectory's right-continuous incumbent at 101
common cost thresholds. No result is available before its first completed
evaluation; coverage is reported and mean accuracy is omitted until all
trajectories in that series have an incumbent. Final incumbents persist after
the evaluation budget is spent. These are compute-truncated fixed-evaluation
policies, not policies optimised for a cost budget. Training GPU-hours exclude
optimiser and LLM inference costs.

The test incumbent is selected by observed validation accuracy, with earliest
selection breaking ties. Only then are test metrics joined offline. Test curves
may decrease as validation improves; taking the best observed test score would
introduce test-based selection and is not done. Proxy replay support is retained:
full-validation discovery and cost remain the offline scoring reference, but
incumbent selection still uses the feedback the optimiser actually observed.

Analyses never pool settings, implementation fingerprints or different LLM
models. Equal replicate numbers alone do not establish pairing. Startup-insensitive
methods copied across sensitivity directories and overlapping seed ranges must
not be counted as independent new trajectories. Existing LLM files have incomplete
serving/token provenance, which is explicitly labelled in comparison metadata.
All uncertainty is conditional on this one fixed seed-42 training landscape;
it does not measure training, dataset or cross-task uncertainty. Baseline design
and sensitivities must be frozen before evaluating their performance.

## Overview

I compare four ways of choosing model configurations in scGeneScope: random search, Bayesian optimization, LLM-guided search using validation metrics, and LLM-guided search using validation metrics plus experiment summaries.

The goal is to test how efficiently each method can find strong configurations when all methods have the same experiment budget.

## Search Space

The search space contains 51 valid model configurations across RNA-only, image-only, and multimodal settings. All 51 are now runnable.

The first (pilot) proxy search comparison was run when only 19 configurations were implemented, so those four trajectories are restricted to that 19-configuration subset. The lookup replays below use all 51 configurations, via:

```python
generate_all_configurations(runnable_only=True)
```

Configurations can vary in model setting, profile setting, RNA encoder, image encoder, aggregation method, and fusion method.

## Search Methods

### Random Search

Random search chooses an untested configuration uniformly from the runnable search space.

It is used as the main non-adaptive baseline.

### Bayesian Search

Bayesian search uses Optuna TPE to choose configurations based on validation results from earlier Bayesian trials.

The TPE parameterization is conditional: the sampler first picks the model setting and profile setting, then only the parameters that apply to that branch (RNA encoder, image encoder, aggregation, fusion). It uses 3 random startup trials. A suggestion that repeats a configuration already tested is discarded and the sampler is asked again.

The optimizer only uses results generated within the Bayesian search condition and does not use results from the Random or LLM search runs.

### LLM Metrics

The LLM Metrics condition uses a fixed language model (`openai_gpt54_mini`) to choose the next experiment.

For each decision, the model receives the configurations and validation results from previous experiments within the LLM Metrics trajectory. Validation accuracy is treated as the main optimization objective, with validation F1 used as supporting information.

The model is asked to balance exploration and exploitation and select one untested configuration from the runnable search space.

### LLM Summary

The LLM Summary condition uses the same search setup as LLM Metrics, but also receives a short scientific summary from each previous experiment in its own trajectory.

Each summary describes what was tested, what the validation result suggests, and what remains uncertain.

This condition tests whether the additional experiment-level context changes the agent's search decisions.

## Experiment Budget

Each search method receives a budget of 10 proxy experiments.

A configuration cannot be repeated within the same search trajectory.

Search methods do not share experimental history with one another.

## Proxy Evaluation

Full scGeneScope training can take several hours for some multi-profile and multimodal configurations. I therefore use a shortened training setup during the sequential search stage.

The proxy setup uses:

- maximum of 10 epochs
- minimum of 1 epoch
- early-stopping patience of 2

The underlying model configuration, data pipeline, validation split, and training code remain the same.

The proxy is used only to provide faster feedback during search.

## Proxy Calibration

Before using the proxy for the search comparison, I ran the shortened setup on the same 10 configurations that had already been evaluated in the full random-search experiments.

Across the 10 matched configurations, the mean absolute difference in validation accuracy was 0.0160.

The Spearman rank correlation between proxy and full validation accuracy was 0.8667 with a p-value of 0.001174.

The proxy also preserved the three highest-performing configurations in the same order.

Based on this comparison, I use proxy validation performance as the feedback signal during search while keeping full training for final model evaluation.

## Full-Training Landscape

All 51 configurations were trained once with the full training setup (seed 42, SLURM array on single L40S GPUs; `full_landscape/`). This gives the full-fidelity validation accuracy of every configuration in the search space and is used as ground truth for scoring the search methods.

The best configuration is multimodal, multiprofile, Transformer aggregation, scVI (200) with weighted ImageNet ViT-L fusion (validation accuracy 0.725). Two configurations stopped at the maximum number of epochs and may be slightly underestimated.

## Proxy Check on All 51 Configurations

With both landscapes complete, I repeated the proxy calibration on all 51 configurations:

- Spearman rank correlation between proxy and full validation accuracy: 0.884 (p = 8.0e-18)
- mean absolute difference in validation accuracy: 0.025

The proxy ranks configurations well overall but does not preserve the top configuration. The proxy's best configuration ranks 5th after full training, and the full-training optimum ranks 4th under the proxy. Transformer-aggregation models are the most underestimated by the short proxy training (up to 0.067 lower). Proxy-based search results should therefore be scored against full-training values.

## Search Objective

Validation accuracy is the primary objective during search.

Validation F1 is recorded as an additional metric but is not the primary optimization target.

The held-out test set is not used to choose experiments.

## Experimental Separation

Search results are stored separately from the original full-training experiments.

- `master_results.csv` contains full-fidelity experiments.
- `proxy_results.csv` contains the proxy calibration runs.
- `proxy_search_results.csv` contains the controlled search comparison.

This keeps the proxy search trajectories separate from the full-training results.

## LLM Decision Logging

For the LLM-guided conditions, the exact prompt and raw model response are saved for each search decision.

The selected configuration and selection reason are also stored with the experiment record.

This makes it possible to review how each LLM decision was made after the search is complete.

## Search Analysis

Search efficiency is measured primarily using best validation accuracy found versus experiment number.

I also compare best validation accuracy against cumulative proxy runtime.

These measurements show both how quickly each method finds a strong configuration in terms of experimental budget and how much training time is required to reach it.

## Lookup Replays

A single trajectory per method cannot separate a better method from a lucky run. Because every configuration now has a known full-training result, each search method is replayed many times without training: when a method selects a configuration, its validation accuracy and F1 are looked up from the full landscape and fed back to the method (`replay_search.py`).

- Each replicate starts from an empty history and uses a different seed (1000 + replicate index).
- Each replicate has a budget of 10 experiments and cannot repeat a configuration.
- Methods use the same selection code as the live search (`random_search.py`, `bayesian_search.py`, `llm_search.py`, `summarize_experiment.py`).
- For each method and experiment number I report the mean best validation accuracy found so far, a 95% bootstrap confidence interval of that mean across replicates (2,000 resamples), the regret to the landscape optimum, the fraction of replicates that have found the optimum or one of the top three configurations, and the cumulative full-training GPU hours.

For the LLM conditions, every prompt and raw response is saved per replicate. The LLM is not seeded, so its replicates differ through sampling in the model.

The replays answer the selection question only. They assume each configuration has one fixed validation score, so they do not include seed-to-seed training noise.

## LLM Prompt Ablation

In the lookup replays, both LLM conditions spent most of their budget on RNA-only configurations and rarely reached the multimodal multiprofile models that hold the top of the landscape. Their selection reasons describe a one-factor-at-a-time ablation starting from the simplest baseline. Three features of the original prompt and setup could cause this:

1. **Candidate order.** Candidates are listed RNA-only first, then image-only, then multimodal, with IDs in that order. Every LLM Metrics replicate started with `config_001`.
2. **Learning objective and unstated budget.** The prompt asks the agent "not only to find a high-performing configuration, but … to learn which design decisions matter", says only that the budget is "limited", and encourages informative experiments "even if [they] may not be the configuration with the highest expected immediate performance".
3. **Isolation principles.** The prompt asks the agent to isolate the effect of one design choice, to avoid changing several choices at once, and not to assume that multimodal, multiprofile or more complex models are better. Moving from an RNA-only to a multimodal configuration changes the image encoder and fusion together.

To test these, I replay the LLM Metrics condition with prompt variants that each change one factor, plus one that changes all of them. The default prompt is unchanged and byte-identical to the one used in the original search.

| Variant | Change from LLM Metrics |
| --- | --- |
| `llm_metrics_shuffled` | Candidates are listed in a random order per replicate, and IDs are assigned in that order, so neither list position nor ID reveals the model setting. History uses the same IDs. |
| `llm_metrics_budget` | The objective paragraph states "experiment k of 10" and the number remaining, and makes the best validation accuracy found within the budget the only objective. The "informative even if not the highest expected performance" principle is removed. |
| `llm_metrics_no_isolation` | The two isolation principles and the "do not assume more complex models are better" principle are removed. |
| `llm_metrics_combined` | All three changes. |
| `llm_metrics_combined_gpu` | All three changes, plus compute awareness: each completed experiment shows its measured training cost in GPU hours, the history starts with the total GPU hours used so far, and the agent is asked to reach the best validation accuracy while keeping total GPU hours low (prefer a cheaper configuration when expected accuracy is similar, but do not avoid an expensive configuration that is likely to be best). Costs of untested configurations are not shown and must be estimated from the architecture. |

Each variant uses the same model, 10 replicates and a budget of 10, and is scored on the full-training landscape. Besides best-found accuracy, I record the share of selections per model setting, how many distinct first picks were made, the mean experiment number of the first multimodal selection, and the number of replicates that never selected a multimodal configuration.

Results are reported in the README (Lookup Replays and LLM Prompt Ablation sections).

## LLM Model Comparison

To test whether the LLM results depend on the model, all seven LLM conditions (LLM Metrics, LLM Summary and the five prompt variants) are replayed with two further models from the Novo GenAI gateway: Claude Opus 5.5 (`anthropic_claude_opus_5_5`) and GPT-6 Astra (`openai_gpt6_astra`), selected with `replay_search.py --llm-model`. Everything else is unchanged: prompts, candidate lists, seeds, 10 replicates, a budget of 10, and scoring on the full-training landscape. The only other difference is the output-token limit for decisions and summaries, which is raised from 500/400 to 2,000 for non-default models because their longer reasons otherwise truncate the JSON response. Results for each model are written to separate `models/<model>/` folders, and `model_comparison.csv` reports every condition after 10 experiments per model.

Results are reported in the README (LLM Model Comparison section).

## Final Evaluation

Proxy performance is not treated as final model performance.

After the search stage is complete, the strongest configurations selected by each search method will be retrained using the full training setup.

The held-out test set will only be evaluated during this final stage.

## Test-Set Evaluation of the Full Landscape

The saved checkpoint of each of the 51 full-training runs is evaluated once on the held-out test set without retraining (`train=false test=true ckpt_path=...`; `full_landscape/submit_test.sh` and `run_test.sbatch`, run on CPU because no GPU capacity was available). Test accuracy and test F1 are merged into `full_landscape_results.csv` by `collect_results.py`. The test results are not used by any search method or replay; they are only used to check how well the validation objective carries over to held-out data (Spearman rank correlation between validation and test accuracy across configurations, and the test rank of the validation optimum). Results are reported in the README (Test-Set Evaluation of the Full Landscape).
