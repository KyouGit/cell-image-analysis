# DDPM Augmentation — Per-class mAP@50 Comparison

> **Experiment G-1/G-5**: Class-conditional DDPM augmentation for rare cell detection  
> Test set: original PBS 25-class (no augmentation)  
> Model: YOLOv11s, 100 epochs, img 640

---

## Per-class mAP@50

| Class | Baseline | Aug_500 | Aug_1000 | TC2000* | Δ (TC2000 − Base) |
|-------|:-------:|:-------:|:--------:|:-------:|:-----------------:|
| Neutrophil | 0.676 | 0.679 | 0.688 | — | — |
| Lymphocyte | 0.994 | 0.994 | 0.994 | — | — |
| Monocyte | 0.982 | 0.981 | 0.980 | — | — |
| Eosinophil | 0.865 | 0.869 | 0.868 | — | — |
| Basophil | 0.814 | 0.809 | 0.810 | — | — |
| **BandNeutrophil** ★ | 0.993 | 0.993 | 0.993 | — | — |
| SegNeutrophil | 0.974 | 0.978 | 0.977 | — | — |
| **hyperSeg** ★ | 0.987 | 0.989 | 0.989 | — | — |
| NormalRBC | 0.967 | 0.969 | 0.962 | — | — |
| Microcyte | 0.936 | 0.927 | 0.927 | — | — |
| Macrocyte | 0.973 | 0.984 | 0.980 | — | — |
| Hypochromia | 0.954 | 0.956 | 0.958 | — | — |
| **Schistocyte** ★ | 0.977 | 0.977 | 0.977 | — | — |
| Spherocyte | 0.651 | 0.670 | 0.667 | — | — |
| **TargetCell** ★ | 0.659 | 0.667 | 0.665 | — | — |
| **Stomatocyte** ★ | 0.951 | 0.952 | 0.949 | — | — |
| Acanthocyte | 0.980 | 0.984 | 0.985 | — | — |
| Echinocyte | 0.826 | 0.819 | 0.818 | — | — |
| **Nucleated** ★ | 0.868 | 0.895 | 0.884 | — | — |
| Blast | 0.777 | 0.753 | 0.777 | — | — |
| Prolymphocyte | 0.797 | 0.788 | 0.796 | — | — |
| **Atypical** ★ | 0.759 | 0.773 | 0.787 | — | — |
| Platelet | 0.954 | 0.956 | 0.963 | — | — |
| GiantPlt | 0.971 | 0.968 | 0.965 | — | — |
| Thrombocytopenia | 0.700 | 0.699 | 0.692 | — | — |
| **mAP@50 (mean)** | **0.8794** | **0.8811** | **0.8821** | **—** | **—** |

★ Rare classes targeted by DDPM augmentation  
\* TC2000: TargetCell-dedicated DDPM (single-class, 1000 epoch, base_ch=96) + 2000 generated images

---

## Rare Class Subset mAP

| | Baseline | Aug_500 | Aug_1000 | TC2000* |
|--|:-------:|:-------:|:--------:|:-------:|
| Rare mAP@50 | 0.8990 | 0.9099 | 0.9069 | — |

Rare classes: BandNeutrophil, hyperSeg, Schistocyte, TargetCell, Stomatocyte, Nucleated, Atypical

---

## Aug_1000 vs Baseline — Improvement Delta

| Class | Δ mAP@50 |
|-------|:--------:|
| Atypical ★ | +0.028 |
| Nucleated ★ | +0.016 |
| Spherocyte | +0.016 |
| Acanthocyte | +0.005 |
| Platelet | +0.009 |
| Macrocyte | +0.007 |
| Hypochromia | +0.004 |
| TargetCell ★ | +0.006 |
| Neutrophil | +0.012 |
| SegNeutrophil | +0.003 |
| hyperSeg ★ | +0.002 |
| BandNeutrophil ★ | -0.000 |
| Schistocyte ★ | -0.000 |
| Stomatocyte ★ | -0.002 |
| Echinocyte | -0.008 |

---

*TC2000 results pending — TargetCell dedicated DDPM training + YOLO retraining in progress.*
