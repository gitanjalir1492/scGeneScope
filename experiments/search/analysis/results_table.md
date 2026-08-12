# Search Results

## Proxy Search Comparison

All search methods use the same 19 runnable configurations and a budget of 10 experiments.

| Method | Trials | Best Proxy Val Acc | Best Proxy Val F1 | Trial Best Found | Best Configuration | Total Proxy Runtime |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Random | 10 | 0.7062 | 0.6720 | 10 | Multimodal, scVI n200 + ImageNet ViT-L, multiprofile, avgpool, concat | 3.10 h |
| Bayesian | In progress | TBD | TBD | TBD | TBD | TBD |
| LLM Metrics | Not started | TBD | TBD | TBD | TBD | TBD |
| LLM Summary | Not started | TBD | TBD | TBD | TBD | TBD |

The main search comparison is based on best validation accuracy found as a function of experiment number. Runtime is also tracked to compare performance against total proxy training time.

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
| Random | TBD | TBD | TBD | TBD | TBD | TBD |
| Bayesian | TBD | TBD | TBD | TBD | TBD | TBD |
| LLM Metrics | TBD | TBD | TBD | TBD | TBD | TBD |
| LLM Summary | TBD | TBD | TBD | TBD | TBD | TBD |

The held-out test set will only be used after the search stage is complete.

## Main Figures

The current analysis script generates:

- `best_validation_vs_trial.png` - best validation accuracy found versus experiment number
- `best_validation_vs_runtime.png` - best validation accuracy found versus cumulative proxy runtime

These figures will update as the Bayesian and LLM search runs finish.
