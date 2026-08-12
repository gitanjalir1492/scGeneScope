# Methods

## Overview

I compare four ways of choosing model configurations in scGeneScope: random search, Bayesian optimization, LLM-guided search using validation metrics, and LLM-guided search using validation metrics plus experiment summaries.

The goal is to test how efficiently each method can find strong configurations when all methods have the same experiment budget.

## Search Space

The target search space contains 51 valid model configurations across RNA-only, image-only, and multimodal settings.

At the time of this study, 19 configurations are fully runnable with the current scGeneScope implementation. The remaining configurations require additional support for ResNet50 combinations, weighted multimodal fusion, or multimodal Transformer aggregation.

All four search methods are restricted to the same 19 runnable configurations using:

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

## Final Evaluation

Proxy performance is not treated as final model performance.

After the search stage is complete, the strongest configurations selected by each search method will be retrained using the full training setup.

The held-out test set will only be evaluated during this final stage.
