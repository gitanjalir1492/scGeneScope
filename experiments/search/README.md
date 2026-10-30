# Autonomous Search Experiments

This folder contains the search experiments I am using to explore the scGeneScope model space.

The main goal is to compare different ways of choosing which model configuration to try next. I am comparing random search, Bayesian optimization, and two LLM-guided search strategies under the same experiment budget.

The LLM experiments test whether an agent can use results from previous experiments to make useful decisions about what to run next.

---

# Files

```text
experiments/search/
│
├── random_search.py
├── bayesian_search.py
├── llm_search.py
├── summarize_experiment.py
├── run_search.py
├── run_experiments.py
├── search_space.py
├── results_io.py
├── analyze_search.py
├── replay_search.py
│
├── full_landscape/
├── proxy/
├── results/
├── logs/
└── runs/
```

## `random_search.py`

Randomly chooses an untested configuration from the runnable search space.

I use this as the baseline for comparing the other search methods.

## `bayesian_search.py`

Uses Optuna TPE to choose experiments based on the validation results from previous Bayesian trials.

This gives me a standard optimization method to compare against both random search and the LLM approaches.

## `llm_search.py`

Uses a fixed LLM (`openai_gpt54_mini` by default) to choose the next experiment. In the lookup replays, other gateway models can be used with `--llm-model` (see [LLM Model Comparison](#llm-model-comparison)).

I am testing two versions.

### Metrics only

The LLM sees the configurations and validation results from its previous experiments:

- validation accuracy
- validation F1
- model configuration

### Metrics + summaries

The LLM gets the same information, but also sees a short summary of what was learned from each previous experiment.

This lets me test whether giving the agent more context about previous results changes the experiments it chooses.

For reproducibility, the prompt and raw response from each LLM decision are also saved.

## `summarize_experiment.py`

Generates the experiment summaries used by the LLM summary condition.

The summary describes what was tested, what the validation results suggest, and what is still uncertain.

## `run_experiments.py`

Runs a selected scGeneScope experiment and records its validation results.

## `run_search.py`

Runs the search loop:

1. choose the next configuration
2. train it
3. record the validation results
4. update the search history
5. repeat until the experiment budget is reached

For the LLM summary condition, a summary is also generated after each completed experiment.

## `search_space.py`

Defines the configurations that can be explored.

The search space contains 51 valid configurations, and all 51 are runnable.

The pilot search comparison below was run when only 19 configurations were implemented, so its four trajectories are restricted to that subset. The lookup replays use all 51.

Every method uses:

```python
generate_all_configurations(runnable_only=True)
```

## `replay_search.py`

Replays the search methods many times on the completed full-training landscape, looking up each selected configuration's validation result instead of training it. See [Lookup Replays](#lookup-replays).

## `full_landscape/`

SLURM scripts that trained all 51 configurations with the full training setup (one array task per configuration). `submit_test.sh` / `run_test.sbatch` evaluate the saved checkpoints on the held-out test set, and `collect_results.py` merges everything into `results/full_landscape_results.csv`.

---

# Search Comparison

I am comparing four search conditions:

| Method | Information used to choose the next experiment |
| --- | --- |
| Random | None |
| Bayesian | Previous validation results |
| LLM Metrics | Previous validation results |
| LLM Summary | Previous validation results + experiment summaries |

Each method gets the same search space, training setup, validation objective, and experiment budget.

The main question is whether the LLM-guided searches can find strong configurations with fewer experiments than random or Bayesian search.

---

# Proxy Experiments

Some of the full scGeneScope experiments take several hours to train, especially the multi-profile and multimodal models. This makes it difficult to run many sequential search experiments.

I therefore use a shorter training run during the search:

- maximum 10 epochs
- minimum 1 epoch
- early stopping patience of 2

Everything else about the configuration and validation setup stays the same.

The proxy score is only used to help the search methods choose experiments. Final configurations will still be trained normally before test evaluation.

## Checking the Proxy

Before using the proxy for the search comparison, I compared it against the 10 full random-search experiments I had already run.

Across those 10 configurations:

- mean absolute validation accuracy error: **0.0160**
- Spearman rank correlation: **0.8667**
- p-value: **0.001174**

The proxy also kept the same top three configurations in the same order.

This was strong enough to use the shorter runs for the search stage while keeping full training for the final evaluation.

## Checking the Proxy on All 51 Configurations

After the full-training landscape finished, I repeated the check on all 51 configurations:

- Spearman rank correlation: **0.884** (p = 8.0e-18)
- mean absolute validation accuracy error: **0.025**

The proxy is good at ranking overall, but it does not keep the top configuration. The proxy's best configuration is 5th after full training, and the true best is 4th under the proxy. Transformer-aggregation models are the most underestimated by short training. Search results are therefore scored on full-training values.

The calibration code is in:

```text
proxy/run_calibration.py
proxy/analyze_calibration.py
```

and the results are stored in:

```text
results/proxy_results.csv
```

---

# Proxy Search Comparison

The main search comparison uses the same proxy setup for all four methods:

- Random
- Bayesian
- LLM Metrics
- LLM Summary

Each method gets:

- the same 19 runnable configurations
- the same 10-epoch maximum
- the same early stopping setup
- the same validation objective
- the same experiment budget
- no access to the held-out test results during search

The 10 Random proxy results come from the proxy calibration runs, so I reuse those results instead of training the same configurations again.

The search results are stored in:

```text
results/proxy_search_results.csv
```

---

# Full-Training Landscape

All 51 configurations were trained once with the full setup (seed 42) using the SLURM scripts in `full_landscape/`. Results are in `results/full_landscape_results.csv`.

Top configurations by validation accuracy:

| Configuration | Val accuracy | Val F1 |
| --- | --- | --- |
| multimodal, multiprofile, Transformer, scVI-200 + weighted ImageNet ViT-L | 0.725 | 0.685 |
| multimodal, multiprofile, Transformer, scVI-200 + concat ImageNet ViT-L | 0.721 | 0.681 |
| multimodal, multiprofile, avgpool, scVI-200 + concat ImageNet ViT-L | 0.715 | 0.687 |
| multimodal, multiprofile, Transformer, scVI-200 + weighted ResNet50 | 0.713 | 0.695 |
| multimodal, multiprofile, avgpool, scVI-200 + weighted ImageNet ViT-L | 0.706 | 0.667 |

The best RNA-only configuration reaches 0.653, and image-only single-profile models are weakest (0.21-0.33).

---

# Lookup Replays

One trajectory per method is not enough to tell a better method from a lucky run. With every configuration already trained, `replay_search.py` reruns each search method many times and looks up the full-training validation result of each selected configuration instead of training it.

Each replicate starts with no history, uses its own seed, has a budget of 10 experiments, and uses the same selection code as the live search.

```bash
python replay_search.py --methods random bayesian --replicates 200
python replay_search.py --methods llm_metrics llm_summary --replicates 10   # needs LLM credentials, see below
python replay_search.py --methods llm_metrics_shuffled llm_metrics_budget llm_metrics_no_isolation llm_metrics_combined llm_metrics_combined_gpu --replicates 10
python replay_search.py --analyze-only
```

The LLM calls go through the Novo GenAI marketplace gateway:

```bash
export OPENAI_API_KEY=$(tr -d '[:space:]' < ~/openai_api_key.txt)
export OPENAI_BASE_URL=https://api.marketplace.novo-genai.com/v1
```

Mean best validation accuracy found (full training), with 95% bootstrap CI across replicates:

| Method | Replicates | After 3 | After 5 | After 10 | Found optimum by 10 | Found a top-3 config by 10 | Mean GPU hours for 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 200 | 0.629 (0.618-0.640) | 0.663 (0.655-0.672) | 0.703 (0.699-0.707) | 24.5% | 52.0% | 40.5 |
| Bayesian (TPE) | 200 | 0.590 (0.580-0.601) | 0.650 (0.641-0.658) | 0.695 (0.691-0.700) | 15.0% | 42.0% | 32.4 |
| LLM Metrics | 10 | 0.534 (0.520-0.555) | 0.652 (0.651-0.653) | 0.653 (0.653-0.653) | 0% | 0% | 9.6 |
| LLM Summary | 10 | 0.586 (0.553-0.619) | 0.653 (0.652-0.653) | 0.659 (0.653-0.667) | 0% | 0% | 19.6 |

With 10 experiments, TPE does slightly worse than random. Its conditional parameterization picks the model setting first, so multimodal configurations (36 of 51, and all of the top ones) are chosen about 50% of the time instead of 71% under uniform random selection, and 10 trials are too few for TPE to correct this.

Both LLM conditions do worse than random. They spent 77-87% of their experiments on RNA-only models. Every LLM Metrics replicate (and 9 of 10 LLM Summary replicates) started with the same configuration (`config_001`, RNA-only single-profile PCA), and all LLM Metrics replicates plateaued at 0.653, the best RNA-only model. Their selection reasons describe a one-factor-at-a-time ablation from the simplest baseline rather than a search for the best model. The replicates are nearly deterministic, so the narrow CIs reflect how similar the trajectories are, not a precise estimate.

## LLM Prompt Ablation

To find out why the LLMs stay on RNA-only models, I replayed LLM Metrics with prompt variants that change one thing each (see METHODS.md for the exact changes). The default prompt is unchanged.

| Variant | Change | After 3 | After 5 | After 10 | Found optimum by 10 | Found a top-3 config by 10 | Multimodal share of picks | Mean GPU hours for 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLM Metrics | none (original prompt) | 0.534 | 0.652 | 0.653 (0.653-0.653) | 0% | 0% | 2% | 9.6 |
| Shuffled | random candidate order and IDs | 0.593 | 0.649 | 0.697 (0.660-0.725) | 80% | 80% | 69% | 34.7 |
| Budget | states "experiment k of 10"; best-found accuracy is the only goal | 0.546 | 0.653 | 0.694 (0.675-0.713) | 40% | 40% | 17% | 14.7 |
| No isolation | removes one-factor-at-a-time and "complex is not better" principles | 0.606 | 0.636 | 0.678 (0.657-0.699) | 20% | 30% | 40% | 32.4 |
| Combined | all three changes | 0.676 | 0.700 | 0.724 (0.722-0.725) | 90% | 100% | 100% | 55.4 |
| Combined + GPU-aware | all three, plus measured GPU hours of completed runs and an instruction to keep total GPU hours low | 0.691 | 0.715 | 0.725 (0.725-0.725) | 100% | 100% | 98% | 50.9 |
| Random (reference) | | 0.629 | 0.663 | 0.703 (0.699-0.707) | 24.5% | 52.0% | 71% | 40.5 |

Each change on its own moves the LLM off the RNA-only plateau, in a different way:

- **Candidate order is the largest single effect.** With the original order every replicate started on `config_001`. Shuffled, the 10 replicates started on 8 different configurations, and 69% of picks were multimodal. The outcome depends on where it starts: 8 replicates found the optimum (on average at experiment 5.5), while 2 stayed on weaker configurations (0.54 and 0.64).
- **Stating the budget makes it exploit at the end.** The agent still explores RNA-only models first, but switches to multimodal models in the last one or two experiments (first multimodal pick at experiment 9.3 on average). That raises the final result from 0.653 to 0.694, and 4 of 10 replicates find the optimum with their last experiment, at modest extra compute (14.7 versus 9.6 GPU hours).
- **Removing the isolation principles** increases multimodal picks (40%), but the agent still starts on RNA-only models and reaches only 0.678 (optimum found in 2 of 10).
- **All three together** pick only multimodal configurations from the first experiment, find a top-3 configuration in every replicate, and find the optimum in 9 of 10 (0.724 mean after 10). It is also expensive (55 GPU hours versus 40 for random), because multimodal multiprofile models take longest to train.
- **Adding GPU-hour awareness** to the combined prompt found the optimum in all 10 replicates (on average at experiment 3.4) and cut the compute for 10 experiments from 55 to 51 GPU hours, by choosing slightly cheaper multimodal configurations (5.1 instead of 5.5 GPU hours per experiment). GPU hours spent until the optimum was found were similar (15.6 versus 16.2 for the combined prompt, among replicates that found it). It still uses more compute than random search (40 GPU hours), because it keeps spending the whole budget on multimodal models after it has found the optimum. Letting the agent stop early would be the next step to save compute.

The original LLM behaviour is therefore mostly caused by the prompt and the candidate ordering, not by an inherent preference of the model for simple configurations. With 10 replicates per variant, these are effect directions rather than precise estimates.

## LLM Model Comparison

To check whether the results depend on the model, I repeated all seven LLM conditions (10 replicates each) with two larger models available through the gateway: Claude Opus 5.5 (`anthropic_claude_opus_5_5`) and GPT-6 Astra (`openai_gpt6_astra`). Prompts, candidates, seeds and scoring are identical; only the model and the output-token limit (2,000 instead of 500/400, because the larger models write longer reasons) change.

```bash
python replay_search.py --llm-model anthropic_claude_opus_5_5 --replicates 10 \
    --methods llm_metrics llm_summary llm_metrics_shuffled llm_metrics_budget llm_metrics_no_isolation llm_metrics_combined llm_metrics_combined_gpu
python replay_search.py --llm-model openai_gpt6_astra --replicates 10 --methods ...   # same seven methods
```

Mean best validation accuracy after 10 experiments / fraction of replicates that found the optimum / mean GPU hours for 10:

| Condition | GPT-5.4 mini | Claude Opus 5.5 | GPT-6 Astra |
| --- | --- | --- | --- |
| LLM Metrics | 0.653 / 0% / 9.6 | 0.717 / 50% / 41.0 | 0.706 / 0% / 21.8 |
| LLM Summary | 0.659 / 0% / 19.6 | 0.707 / 0% / 67.9 | 0.719 / 70% / 24.7 |
| Shuffled | 0.697 / 80% / 34.7 | 0.710 / 30% / 50.6 | 0.720 / 50% / 22.4 |
| Budget | 0.694 / 40% / 14.7 | 0.717 / 10% / 46.9 | 0.725 / 100% / 21.7 |
| No isolation | 0.678 / 20% / 32.4 | 0.725 / 100% / 35.9 | 0.722 / 60% / 84.9 |
| Combined | 0.724 / 90% / 55.4 | 0.725 / 100% / 38.2 | 0.725 / 100% / 39.8 |
| Combined + GPU-aware | 0.725 / 100% / 50.9 | 0.725 / 100% / 38.7 | 0.725 / 100% / 41.5 |
| Random (reference) | 0.703 / 24.5% / 40.5 | | |

- **Both larger models beat GPT-5.4 mini and random search under the original prompts**, but they still follow the prompt's ablation strategy: every replicate of LLM Metrics and LLM Summary starts on `config_001`, and the first multimodal pick comes late (experiment 4.6-7.0 on average). Opus reaches multimodal models in time to find the optimum in 5 of 10 LLM Metrics replicates; every GPT-6 Astra LLM Metrics replicate plateaus at the 5th-best configuration (0.706).
- **The prompt fixes matter less for the larger models but still help.** With the combined prompt all three models find the optimum in (almost) every replicate, so the model choice mostly changes cost: Opus and GPT-6 Astra need 38-42 GPU hours, about the same as random search, versus 51-55 for GPT-5.4 mini.
- **The models respond differently to single changes.** Stating the budget is enough for GPT-6 Astra (optimum in 10 of 10 at 21.7 GPU hours, the cheapest condition that reliably finds it), but not for Opus (1 of 10). Removing the isolation principles is enough for Opus (10 of 10) but makes GPT-6 Astra pick almost only multimodal configurations (98%), the most expensive condition (84.9 GPU hours).
- **Opus uses the most compute** in four of the five conditions without combined changes (41-68 GPU hours); GPT-6 Astra stays at 22-25 GPU hours in all of them except No isolation.

As before, 10 replicates per cell give effect directions, not precise estimates.

Outputs:

```text
results/lookup_replays/<method>.csv                     one row per replicate x trial
results/lookup_replays/llm_logs/                        prompts and raw LLM responses per replicate
analysis/lookup_replay/replay_curves.csv                per method x trial: mean, 95% CI, regret, P(found best / top-3), GPU hours
analysis/lookup_replay/replay_selection_mix.csv         per method: share of RNA-only / image-only / multimodal picks, first picks
analysis/lookup_replay/replay_best_validation.png       Random vs Bayesian vs LLM Metrics vs LLM Summary
analysis/lookup_replay/replay_llm_prompt_ablation.png   LLM prompt variants vs Random
results/lookup_replays/models/<model>/                  the same replay files for a non-default --llm-model
analysis/lookup_replay/models/<model>/                  the same analysis files for a non-default --llm-model
analysis/lookup_replay/model_comparison.{csv,png}       all LLM conditions after 10 experiments, per model
```

---

# Results Files

### `results/master_results.csv`

Full training experiments.

### `results/proxy_results.csv`

The proxy runs used to compare short training against full training.

### `results/proxy_search_results.csv`

The experiments used for the Random vs Bayesian vs LLM search comparison.

### `results/proxy_landscape_results.csv`

Proxy training of all 51 configurations.

### `results/full_landscape_results.csv`

Full training of all 51 configurations, with epochs trained, runtime, SLURM metadata, and held-out test accuracy and F1 (`test_accuracy`, `test_f1`).

### `results/lookup_replays/`

Repeated lookup replays of each search method.

---

# Analysis

`analyze_search.py` compares the search methods as experiments finish.

It produces:

```text
analysis/
├── search_curves.csv
├── search_summary.csv
├── best_validation_vs_trial.png
└── best_validation_vs_runtime.png
```

The main comparison is best validation accuracy found versus number of experiments run. I also track performance against total proxy training time.

---

# Final Evaluation

The proxy runs are for choosing configurations, not for reporting final model performance.

After the search experiments are complete, the strongest configurations found by each method will be trained using the full training setup. The held-out test set will only be used for this final evaluation.

## Test-Set Evaluation of the Full Landscape

All 51 full-training checkpoints were evaluated once on the held-out test set (no retraining; `full_landscape/submit_test.sh`, run on CPU). The test results were not used by any search method.

Test accuracy is much lower than validation accuracy (typically by 0.15-0.25), and the two rank configurations only moderately alike (Spearman 0.52 across the 51 configurations).

| Configuration | Val accuracy (rank) | Test accuracy (rank) |
| --- | --- | --- |
| multimodal, multiprofile, Transformer, scVI-200 + weighted ImageNet ViT-L | 0.725 (1) | 0.498 (12) |
| multimodal, multiprofile, Transformer, scVI-200 + concat ImageNet ViT-L | 0.721 (2) | 0.493 |
| multimodal, multiprofile, avgpool, scVI-200 + concat ImageNet ViT-L | 0.715 (3) | 0.539 |
| multimodal, multiprofile, Transformer, scVI-200 + weighted ResNet50 | 0.713 (4) | 0.545 (4) |
| multimodal, multiprofile, avgpool, scVI-200 + weighted ImageNet ViT-L | 0.706 (5) | 0.549 (3) |
| multimodal, multiprofile, avgpool, scVI-200 + weighted ResNet50 | 0.678 (7) | **0.589 (1)** |
| RNA-only, multiprofile, avgpool, scVI-200 | 0.650 (9) | 0.561 (2) |

The validation optimum, which the search methods are scored on, is only 12th on the test set, and avgpool models generally hold up better on test than Transformer-aggregation models. Finding the validation optimum faster therefore does not translate directly into a better test result; the search comparison above measures search efficiency on the validation objective only.
