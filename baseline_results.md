# scGeneScope Baseline Results

This document tracks the baseline experiments reproduced from the scGeneScope benchmark.

## Summary

| Modality | Profile Type | Embedding | Status | Best Val Acc | Val F1 | Notes |
|----------|--------------|-----------|--------|-------------:|-------:|-------|
| RNA-seq | Single-profile | PCA (2000D) | ✅ Complete | 0.4575 | 0.4362 | 10 epochs, CPU |

---

## Experiment 1

**Configuration**

- Experiment: `rnaseq/singleprofile/train_on_pca_n2000`
- Date: July 17, 2026
- Hardware: CPU
- Epochs: 10
- Batch size: 256
- Seed: 23456

**Results**

| Metric | Value |
|--------|------:|
| Best validation accuracy | **0.4575** |
| Validation F1 | **0.4362** |
| Validation loss | **2.1469** |
| Best epoch | **7** |

**Checkpoint**

```
logs/train/runs/2026-07-17_13-51-28/checkpoints/epoch=7-step=3248.ckpt
```

**Notes**

- First successful end-to-end benchmark reproduction.
- Trained for 10 epochs on CPU.
- Held-out test set has not been evaluated yet.
