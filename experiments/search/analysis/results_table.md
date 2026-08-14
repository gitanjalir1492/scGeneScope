# Search Results

## Proxy Search Comparison

All search methods use the same 19 runnable configurations and a budget of 10 experiments.

| Method | Trials | Best Proxy Val Acc | Best Proxy Val F1 | Best Experiment | Total Proxy Runtime |
| --- | ---: | ---: | ---: | --- | ---: |
| Random | 10 | 0.7062 | 0.6720 | random_010 | 3.10 h |
| Bayesian | 10 | 0.7159 | 0.6766 | bayes_010 | 3.07 h |
| LLM Metrics | 10 | 0.7038 | 0.6656 | llm_metrics_010 | 1.98 h |
| LLM Summary | 10 | 0.7148 | 0.6704 | llm_summary_010 | 2.77 h |

Bayesian search achieved the highest validation accuracy in the 10-trial pilot at 0.7159. LLM Summary reached a similar best validation accuracy of 0.7148, while LLM Metrics reached 0.7038 and Random reached 0.7062.

The main search comparison is based on best validation accuracy found as a function of experiment number. Runtime is also tracked to compare performance against cumulative proxy training time.

## Proxy Calibration

The shortened training setup was checked against the 10 full random-search experiments before being used for the search comparison.

| Metric | Result |
| --- | ---: |
| Matched configurations | 10 |
| Mean absolute validation accuracy error | 0.0160 |
| Spearman rank correlation | 0.8667 |
| Spearman p-value | 0.001174 |
| Top three configurations preserved in order | Yes |

These results support using the proxy as a faster signal for choosing experiments. Final model performance will still be measured using full training.

## Full-Fidelity Confirmation

The strongest configurations selected during proxy search will be retrained using the normal training setup.

| Search Method | Selected Configuration | Proxy Val Acc | Full Val Acc | Full Val F1 | Test Acc | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random | TBD | 0.7062 | TBD | TBD | TBD | TBD |
| Bayesian | TBD | 0.7159 | TBD | TBD | TBD | TBD |
| LLM Metrics | TBD | 0.7038 | TBD | TBD | TBD | TBD |
| LLM Summary | TBD | 0.7148 | TBD | TBD | TBD | TBD |

The held-out test set will only be used after the search stage is complete.

## Main Figures

The analysis script generates:

- `best_validation_vs_trial.png` - best validation accuracy found versus experiment number
- `best_validation_vs_runtime.png` - best validation accuracy found versus cumulative proxy runtime

The current figures contain the completed 10-trial Random, Bayesian, LLM Metrics, and LLM Summary pilot searches.
