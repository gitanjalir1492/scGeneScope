# scGeneScope Baseline Results

This document summarizes the scGeneScope baselines reproduced for this project. The goal is to establish a baseline frontier across RNA-only, image-only, multiprofile, and multimodal models before moving on to search space exploration and agentic optimization.

---

## Baseline Selection

The scGeneScope repository contains many experiment configurations. Rather than reproducing every available model, I selected a representative set of the strongest and most relevant baselines based on the project objectives and discussions with Marc.

The selected experiments span the four primary modeling settings used throughout this project: RNA-only, image-only, multiprofile, and multimodal. Together, these models establish a baseline frontier that will serve as the reference point for future search-space exploration and agent-generated experiments.

---

## Baseline Progress

| Category | Model | Status | Best Val Acc | Test Acc | Test F1 |
|----------|-------|:------:|-------------:|---------:|--------:|
| RNA-only | PCA (n2000) | ✅ | 0.487 | 0.261 | 0.266 |
| RNA-only | scVI (n200) | ✅ | 0.524 | 0.522 | 0.470 |
| RNA-only | scGPT | ✅ | 0.405 | 0.384 | 0.375 |
| Image-only | ImageNet ViT-L | ✅ | 0.224 | 0.195 | 0.179 |
| Image-only | ResNet50 | ✅ | 0.207 | 0.214 | 0.189 |
| RNA Multiprofile | Transformer + scVI | ✅ | 0.652 | 0.543 | 0.522 |
| Image Multiprofile | AvgPool + ViT-L | ✅ | 0.359 | 0.267 | 0.240 |
| Multimodal | scVI + ViT-H | ✅ | 0.531 | 0.523 | 0.517 |
| Multimodal Multiprofile | AvgPool + scVI + ViT-H | ⬜ | - | - | - |

---

## Experiment 1 — RNA-only: PCA (n2000)

**Experiment**

`rnaseq/singleprofile/train_on_pca_n2000`

| Metric | Value |
|--------|------:|
| Best Val Acc | **0.487** |
| Test Acc | **0.261** |
| Test F1 | **0.266** |
| Test Loss | **3.8361** |
| Best Epoch | **9** |

**Run**

```text
logs/train/runs/2026-07-28_11-41-11
```

---

## Experiment 2 — Image-only: ImageNet ViT-L

**Experiment**

`imaging/singleprofile/train_on_concat_imagenet_vit_l`

| Metric | Value |
|--------|------:|
| Best Val Acc | **0.224** |
| Test Acc | **0.195** |
| Test F1 | **0.179** |
| Test Loss | **4.0733** |
| Best Epoch | **5** |

**Run**

```text
logs/train/runs/2026-07-28_11-24-04
```

---

## Experiment 3 — RNA-only: scVI (n200)

**Experiment**

`rnaseq/singleprofile/train_on_scvi_n200`

| Metric | Value |
|--------|------:|
| Best Val Acc | **0.524** |
| Test Acc | **0.522** |
| Test F1 | **0.470** |
| Test Loss | **2.0509** |
| Best Epoch | **3** |

**Run**

```text
logs/train/runs/2026-07-28_12-36-20
```

---

## Experiment 4 — RNA-only: scGPT

**Experiment**

`rnaseq/singleprofile/train_on_scgpt`

| Metric | Value |
|--------|------:|
| Best Val Acc | **0.405** |
| Test Acc | **0.384** |
| Test F1 | **0.375** |
| Test Loss | **2.4610** |
| Best Epoch | **25** |

**Run**

```text
logs/train/runs/2026-07-28_12-36-52
```

---

## Experiment 5 — Image-only: ResNet50

**Experiment**

`imaging/singleprofile/train_on_concat_resnet50`

| Metric | Value |
|--------|------:|
| Best Val Acc | **0.207** |
| Test Acc | **0.214** |
| Test F1 | **0.189** |
| Test Loss | **4.0500** |
| Best Epoch | **14** |

**Run**

```text
logs/train/runs/2026-07-28_12-37-23
```

---

## Experiment 6 — RNA Multiprofile: Transformer + scVI

**Experiment**

`rnaseq/multiprofile/train_transformerpool_on_scvi_n200`

| Metric | Value |
|--------|------:|
| Best Val Acc | **0.652** |
| Test Acc | **0.543** |
| Test F1 | **0.522** |
| Test Loss | **3.7142** |
| Best Epoch | **10** |

**Run**

```text
logs/train/runs/2026-07-28_12-46-02
```

---

## Experiment 7 — Multimodal: scVI + ViT-H

**Experiment**

`multimodal/singleprofile/train_on_scvi_n200_with_concat_imagenet_vit_h`

| Metric | Value |
|--------|------:|
| Best Val Acc | **0.531** |
| Test Acc | **0.523** |
| Test F1 | **0.517** |
| Test Loss | **1.8429** |
| Best Epoch | **7** |

**Run**

```text
logs/train/runs/2026-07-28_13-32-37
```

---

## Experiment 8 — Image Multiprofile: AvgPool + ViT-L

**Experiment**

`imaging/multiprofile/train_avgpool_on_concat_imagenet_vit_l`

| Metric | Value |
|--------|------:|
| Best Val Acc | **0.359** |
| Test Acc | **0.267** |
| Test F1 | **0.240** |
| Test Loss | **2.8174** |
| Best Epoch | **14** |

**Run**

```text
logs/train/runs/2026-07-29_09-54-54
```

---

## Baseline Observations

- The reproduced baselines closely match the performance trends reported in the scGeneScope paper.

- RNA-based models consistently outperform image-only models.

- Among the RNA-only models, **scVI** achieved the strongest performance.

- Multiprofile learning provides a clear improvement over single-profile models for both RNA and image embeddings.

- The image multiprofile model improves over the image single-profile baseline, consistent with the findings reported in the original paper.

- Most reproduced results are within a few percentage points of the reported paper values, confident that the baseline pipeline has been successfully reproduced!

- The multimodal multiprofile baseline is the final remaining experiment needed to complete the baseline frontier before moving to search-space exploration and agentic optimization.
