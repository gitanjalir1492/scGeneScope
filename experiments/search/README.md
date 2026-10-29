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

The search space contains 51 valid configurations, and all 51 are runnable.

The pilot search comparison below was run when only 19 configurations were implemented, so its four trajectories are restricted to that subset. The lookup replays use all 51.

Every method uses:

```python
generate_all_configurations(runnable_only=True)
```

## `replay_search.py`

Replays the search methods many times on the completed full-training landscape, looking up each selected configuration's validation result instead of training it. See [Lookup Replays](#lookup-replays).

## `full_landscape/`

SLURM scripts that trained all 51 configurations with the full training setup (one array task per configuration).

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
python replay_search.py --methods llm_metrics llm_summary --replicates 5   # needs OpenAI credentials
python replay_search.py --analyze-only
```

Results with 200 replicates per method (mean best validation accuracy found, 95% bootstrap CI):

| Method | After 3 | After 5 | After 10 | Found optimum by 10 | Found a top-3 config by 10 |
| --- | --- | --- | --- | --- | --- |
| Random | 0.629 (0.618-0.640) | 0.663 (0.655-0.672) | 0.703 (0.699-0.707) | 24.5% | 52.0% |
| Bayesian (TPE) | 0.590 (0.580-0.601) | 0.650 (0.641-0.658) | 0.695 (0.691-0.700) | 15.0% | 42.0% |

With 10 experiments, TPE does slightly worse than random. Its conditional parameterization picks the model setting first, so multimodal configurations (36 of 51, and all of the top ones) are chosen about 50% of the time instead of 71% under uniform random selection, and 10 trials are too few for TPE to correct this.

The LLM conditions have not been replayed yet; they need OpenAI API access.

Outputs:

```text
results/lookup_replays/<method>.csv        one row per replicate x trial
results/lookup_replays/llm_logs/           prompts and raw LLM responses per replicate
analysis/lookup_replay/replay_curves.csv   per method x trial: mean, 95% CI, regret, P(found best / top-3), GPU hours
analysis/lookup_replay/replay_best_validation.png
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

Full training of all 51 configurations, with epochs trained, runtime and SLURM metadata.

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
