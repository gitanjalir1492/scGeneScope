# Autonomous Search Framework

This directory contains the autonomous search framework used to compare different strategies for exploring the scGeneScope model search space.

The goal is to compare how efficiently different search strategies identify high-performing multimodal cellular profiling models under an identical experimental budget.

---

# Directory Overview

```
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
│
├── results/
│   ├── master_results.csv
│   └── backups/
│
├── logs/
└── runs/
```

---

# Search Methods

## random_search.py

Selects an untested runnable configuration uniformly at random.

Purpose:

- Random-search baseline
- Reference point for evaluating more intelligent search strategies

---

## bayesian_search.py

Uses Bayesian Optimization (Optuna) to select the next experiment using previous validation performance.

Purpose:

- Learn which regions of the search space appear promising
- Balance exploration and exploitation

---

## llm_search.py

Uses a fixed language model (`openai_gpt54_mini`) to choose the next experiment.

Two memory conditions are supported.

### Metrics Memory

The model only receives previous validation metrics.

```
Validation Accuracy
Validation F1
Configuration
```

### Summary Memory

The model receives both validation metrics and scientific summaries generated from previous experiments.

```
Validation Accuracy
Validation F1
Experiment Summary
```

This allows comparison between numerical memory and higher-level scientific memory.

---

## summarize_experiment.py

Generates a concise scientific summary after each completed `llm_summary` experiment.

Each summary describes:

- what was tested
- what the validation results suggest
- remaining uncertainty

Summaries are stored inside

```
results/master_results.csv
```

and become memory for future LLM decisions.

---

# Execution Scripts

## run_experiments.py

Runs a planned experiment.

Responsibilities:

- launches scGeneScope training
- records validation metrics
- updates the shared experiment table

---

## run_search.py

Runs a complete autonomous search loop.

Workflow:

1. Select next experiment
2. Train model
3. Record validation metrics
4. Generate experiment summary (LLM Summary only)
5. Repeat until the experiment budget is reached

The search loop includes:

- dry-run mode
- GPU availability checks
- automatic experiment summaries
- automatic experiment bookkeeping

---

# Search Space

## search_space.py

Defines every runnable configuration.

Each configuration specifies:

- model setting
- profile setting
- RNA representation
- image representation
- aggregation method
- fusion method

Every configuration receives a deterministic configuration ID so that all search methods evaluate exactly the same search space.

---

# Results

```
results/
```

Contains the shared experiment table used by every search strategy.

```
master_results.csv
```

Stores

- experiment metadata
- validation metrics
- test metrics
- experiment summaries
- execution information

```
backups/
```

Contains timestamped backups created before search resets.

---

# Logs

```
logs/
```

Contains training logs produced during experiment execution.

---

# Runs

```
runs/
```

Contains Hydra output directories for completed training runs.

---

# Typical Commands

Generate one random proposal

```bash
python random_search.py
```

Generate one Bayesian proposal

```bash
python bayesian_search.py
```

Generate one LLM proposal

```bash
python llm_search.py \
    --memory-mode metrics \
    --call-api
```

Preview an autonomous search

```bash
python run_search.py \
    --method llm_metrics \
    --budget 5 \
    --gpu 2
```

Execute an autonomous search

```bash
python run_search.py \
    --method llm_metrics \
    --budget 5 \
    --gpu 2 \
    --execute
```

---

# Search Strategies

| Method | Memory | Decision Mechanism |
|----------|--------|--------------------|
| Random | None | Uniform random sampling |
| Bayesian | Validation metrics | Bayesian optimization (Optuna) |
| LLM Metrics | Validation metrics | Fixed LLM reasoning |
| LLM Summary | Validation metrics + scientific summaries | Fixed LLM reasoning with structured scientific memory |

All search methods evaluate the same search space, use the same training pipeline, and are compared under an identical experimental budget.

The objective is to determine whether LLM-guided sequential experimentation can discover stronger model configurations more efficiently than conventional search strategies.
