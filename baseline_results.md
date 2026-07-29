# scGeneScope Baseline Reproduction

This document summarizes the baseline models reproduced from the scGeneScope repository. These experiments establish a reference frontier across RNA-only, image-only, multiprofile, and multimodal models before beginning search-space exploration and agentic optimization.

---

## Baseline Summary

| Category | Model | Published Acc | Reproduced Acc | Δ Acc | Test F1 | Status |
|----------|-------|----------:|---------------:|------:|--------:|:------:|
| RNA-only | PCA (n2000) | 0.207 | **0.261** | **+0.054** | 0.266 | ✅ |
| RNA-only | scVI (n200) | 0.516 | **0.522** | **+0.006** | 0.470 | ✅ |
| RNA-only | scGPT | 0.381 | **0.384** | **+0.003** | 0.375 | ✅ |
| Image-only | ImageNet ViT-L | 0.208 | **0.195** | **−0.013** | 0.179 | ✅ |
| Image-only | ResNet50 | 0.213 | **0.214** | **+0.001** | 0.189 | ✅ |
| RNA Multiprofile | Transformer + scVI | 0.456 | **0.543** | **+0.087** | 0.522 | ✅ |
| Image Multiprofile | AvgPool + ViT-L | 0.258 | **0.267** | **+0.009** | 0.240 | ✅ |
| Multimodal | scVI + ViT-H | 0.526 | **0.523** | **−0.003** | 0.517 | ✅ |
| Multimodal Multiprofile | AvgPool + scVI + ViT-H | 0.587 | - | - | - | ⬜ |

---

## Reproduced Runs

| Category | Experiment | Run Directory |
|----------|------------|---------------|
| RNA-only | `rnaseq/singleprofile/train_on_pca_n2000` | `logs/train/runs/2026-07-28_11-41-11` |
| RNA-only | `rnaseq/singleprofile/train_on_scvi_n200` | `logs/train/runs/2026-07-28_12-36-20` |
| RNA-only | `rnaseq/singleprofile/train_on_scgpt` | `logs/train/runs/2026-07-28_12-36-52` |
| Image-only | `imaging/singleprofile/train_on_concat_imagenet_vit_l` | `logs/train/runs/2026-07-28_11-24-04` |
| Image-only | `imaging/singleprofile/train_on_concat_resnet50` | `logs/train/runs/2026-07-28_12-37-23` |
| RNA Multiprofile | `rnaseq/multiprofile/train_transformerpool_on_scvi_n200` | `logs/train/runs/2026-07-28_12-46-02` |
| Image Multiprofile | `imaging/multiprofile/train_avgpool_on_concat_imagenet_vit_l` | `logs/train/runs/2026-07-29_09-54-54` |
| Multimodal | `multimodal/singleprofile/train_on_scvi_n200_with_concat_imagenet_vit_h` | `logs/train/runs/2026-07-28_13-32-37` |

---

## Key Observations

- Eight of the nine planned baseline models have been successfully reproduced.
- Reproduced performance closely matches the published scGeneScope results across nearly all baselines.
- **scVI** is the strongest RNA-only baseline.
- Multiprofile learning improves performance over single-profile models for both RNA and image embeddings.
- The RNA multiprofile model exceeded the published performance by **+0.087** test accuracy and will be investigated further.
- The multimodal multiprofile baseline is the final remaining experiment before beginning search-space exploration and agentic optimization.

---
