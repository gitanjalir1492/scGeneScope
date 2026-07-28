# scGeneScope Baseline Results

This document tracks the baseline experiments reproduced for the scGeneScope benchmark. The goal is to establish a representative set of RNA-only, image-only, multiprofile, and multimodal baselines before moving on to search space exploration and agentic optimization experiments.

---

## Baseline Selection

The scGeneScope repository includes many different model configurations. Rather than reproducing every available experiment, I selected a representative set based on the project brief and the original scGeneScope paper. These experiments cover the major model categories while keeping the number of runs manageable.

Within each category, I prioritized models that either performed well in the original paper or represented a different modeling approach. Additional baselines can be added later if they become useful for expanding the search space or supporting further analysis.

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
| Image Multiprofile | AvgPool + ViT-L | ⬜ | - | - | - |
| Multimodal | scVI + ViT-H | ✅ | 0.531 | 0.523 | 0.517 |
| Multimodal Multiprofile | AvgPool + scVI + ViT-H | ⬜ | - | - | - |

---

## Experiment 1

### RNA-only: PCA (n2000)

**Configuration**

- Experiment: `rnaseq/singleprofile/train_on_pca_n2000`
- Date: 2026-07-28
- Hardware: GPU (RTX 6000)

**Results**

| Metric | Value |
|--------|------:|
| Best validation accuracy | **0.487** |
| Test accuracy | **0.261** |
| Test F1 | **0.266** |
| Test loss | **3.8361** |
| Best epoch | **9** |

**Run directory**

```text
logs/train/runs/2026-07-28_11-41-11
```

---

## Experiment 2

### Image-only: ImageNet ViT-L

**Configuration**

- Experiment: `imaging/singleprofile/train_on_concat_imagenet_vit_l`
- Date: 2026-07-28
- Hardware: GPU (RTX 6000)

**Results**

| Metric | Value |
|--------|------:|
| Best validation accuracy | **0.224** |
| Test accuracy | **0.195** |
| Test F1 | **0.179** |
| Test loss | **4.0733** |
| Best epoch | **5** |

**Run directory**

```text
logs/train/runs/2026-07-28_11-24-04
```

---

## Experiment 3

### RNA-only: scVI (n200)

**Configuration**

- Experiment: `rnaseq/singleprofile/train_on_scvi_n200`
- Date: 2026-07-28
- Hardware: GPU (RTX 6000)

**Results**

| Metric | Value |
|--------|------:|
| Best validation accuracy | **0.524** |
| Test accuracy | **0.522** |
| Test F1 | **0.470** |
| Test loss | **2.0509** |
| Best epoch | **3** |

**Run directory**

```text
logs/train/runs/2026-07-28_12-36-20
```

---

## Experiment 4

### RNA-only: scGPT

**Configuration**

- Experiment: `rnaseq/singleprofile/train_on_scgpt`
- Date: 2026-07-28
- Hardware: GPU (RTX 6000)

**Results**

| Metric | Value |
|--------|------:|
| Best validation accuracy | **0.405** |
| Test accuracy | **0.384** |
| Test F1 | **0.375** |
| Test loss | **2.4610** |
| Best epoch | **25** |

**Run directory**

```text
logs/train/runs/2026-07-28_12-36-52
```

---

## Experiment 5

### Image-only: ResNet50

**Configuration**

- Experiment: `imaging/singleprofile/train_on_concat_resnet50`
- Date: 2026-07-28
- Hardware: GPU (RTX 6000)

**Results**

| Metric | Value |
|--------|------:|
| Best validation accuracy | **0.207** |
| Test accuracy | **0.214** |
| Test F1 | **0.189** |
| Test loss | **4.0500** |
| Best epoch | **14** |

**Run directory**

```text
logs/train/runs/2026-07-28_12-37-23
```

---

## Experiment 6

### RNA Multiprofile: Transformer + scVI

**Configuration**

- Experiment: `rnaseq/multiprofile/train_transformerpool_on_scvi_n200`
- Date: 2026-07-28
- Hardware: GPU (RTX 6000)

**Results**

| Metric | Value |
|--------|------:|
| Best validation accuracy | **0.652** |
| Test accuracy | **0.543** |
| Test F1 | **0.522** |
| Test loss | **3.7142** |
| Best epoch | **10** |

**Run directory**

```text
logs/train/runs/2026-07-28_12-46-02
```

---

## Experiment 7

### Multimodal Single-Profile: scVI + ViT-H

**Configuration**

- Experiment: `multimodal/singleprofile/train_on_scvi_n200_with_concat_imagenet_vit_h`
- Date: 2026-07-28
- Hardware: GPU (RTX 6000)

**Results**

| Metric | Value |
|--------|------:|
| Best validation accuracy | **0.531** |
| Test accuracy | **0.523** |
| Test F1 | **0.517** |
| Test loss | **1.8429** |
| Best epoch | **7** |

**Run directory**

```text
logs/train/runs/2026-07-28_13-32-37
```
