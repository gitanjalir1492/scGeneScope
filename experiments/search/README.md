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
│
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

Uses a fixed LLM (`openai_gpt54_mini`) to choose the next experiment.

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

The full target search space contains 51 valid configurations. Right now, 19 of these are runnable with the implementations currently available in scGeneScope.

The remaining configurations need additional support for ResNet50 combinations, weighted multimodal fusion, or multimodal Transformer aggregation.

For the search comparison, every method uses:

```python
generate_all_configurations(runnable_only=True)
```

so they all choose from the same 19 configurations.

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

# Results Files

### `results/master_results.csv`

Full training experiments.

### `results/proxy_results.csv`

The proxy runs used to compare short training against full training.

### `results/proxy_search_results.csv`

The experiments used for the Random vs Bayesian vs LLM search comparison.

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
