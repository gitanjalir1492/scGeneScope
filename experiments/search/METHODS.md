# Methods

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

The LLM Metrics condition uses a fixed language model to choose the next experiment.

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

Each variant uses the same model, 5 replicates and a budget of 10, and is scored on the full-training landscape. Besides best-found accuracy, I record the share of selections per model setting, how many distinct first picks were made, the mean experiment number of the first multimodal selection, and the number of replicates that never selected a multimodal configuration.

Results are reported in the README (Lookup Replays and LLM Prompt Ablation sections).

## Final Evaluation

Proxy performance is not treated as final model performance.

After the search stage is complete, the strongest configurations selected by each search method will be retrained using the full training setup.

The held-out test set will only be evaluated during this final stage.
