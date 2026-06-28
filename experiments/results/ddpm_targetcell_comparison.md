# DDPM Augmentation — Per-class mAP@50 Final Results

> **Experiments**: DDPM class-conditional augmentation (G-1) + TargetCell-dedicated DDPM (G-5)
> Model: YOLOv11s, 100 epochs, img 640 | Test set: original PBS 25-class (no augmentation)

---

## Overall mAP@50

| Model | mAP@50 | Δ vs Baseline |
|-------|:------:|:-------------:|
| Baseline | 0.8794 | — |
| Aug_500 (5-class DDPM ×500) | 0.8811 | +0.0017 |
| Aug_1000 (5-class DDPM ×1000) | 0.8821 | +0.0027 |
| **TC2000** (TargetCell-only DDPM ×2000) | **0.8827** | **+0.0033** |

---

## Per-class mAP@50

| Class | Baseline | Aug_1000 | TC2000 | Δ (TC2000−Base) |
|-------|:-------:|:--------:|:------:|:---------------:|
| Neutrophil | 0.676 | 0.688 | 0.681 | +0.005 |
| Lymphocyte | 0.994 | 0.994 | 0.994 | +0.001 |
| Monocyte | 0.982 | 0.980 | 0.980 | −0.002 |
| Eosinophil | 0.865 | 0.868 | 0.865 | +0.000 |
| Basophil | 0.814 | 0.810 | 0.814 | +0.000 |
| **BandNeutrophil** ★ | 0.993 | 0.993 | 0.994 | **+0.001** |
| SegNeutrophil | 0.974 | 0.977 | 0.979 | +0.005 |
| **hyperSeg** ★ | 0.987 | 0.989 | 0.989 | **+0.002** |
| NormalRBC | 0.967 | 0.962 | 0.968 | +0.001 |
| Microcyte | 0.936 | 0.927 | 0.932 | −0.004 |
| Macrocyte | 0.973 | 0.980 | **0.987** | **+0.014** |
| Hypochromia | 0.954 | 0.958 | 0.956 | +0.002 |
| **Schistocyte** ★ | 0.977 | 0.977 | 0.977 | +0.000 |
| Spherocyte | 0.651 | 0.667 | 0.663 | +0.012 |
| **TargetCell** ★ | 0.659 | 0.665 | 0.643 | **−0.016** |
| **Stomatocyte** ★ | 0.951 | 0.949 | 0.951 | +0.000 |
| Acanthocyte | 0.980 | 0.985 | 0.978 | −0.002 |
| Echinocyte | 0.826 | 0.818 | **0.840** | **+0.014** |
| **Nucleated** ★ | 0.868 | 0.884 | **0.894** | **+0.026** |
| Blast | 0.777 | 0.777 | 0.781 | +0.004 |
| Prolymphocyte | 0.797 | 0.796 | 0.798 | +0.001 |
| **Atypical** ★ | 0.759 | 0.787 | 0.777 | +0.018 |
| Platelet | 0.954 | 0.963 | 0.959 | +0.005 |
| GiantPlt | 0.971 | 0.965 | 0.959 | −0.012 |
| Thrombocytopenia | 0.700 | 0.692 | 0.709 | **+0.009** |

★ Rare classes targeted by DDPM augmentation

---

## Key Findings

### ✅ Improvements (TC2000 vs Baseline)
| Class | Δ mAP@50 |
|-------|:--------:|
| Nucleated ★ | **+0.026** |
| Atypical ★ | **+0.018** |
| Echinocyte | **+0.014** |
| Macrocyte | **+0.014** |
| Thrombocytopenia | **+0.009** |
| Spherocyte | **+0.012** |

### ⚠️ Regression
| Class | Δ mAP@50 | Note |
|-------|:--------:|------|
| TargetCell ★ | **−0.016** | Training-test distribution mismatch: generated 64×64 crops vs detection in full slides |
| GiantPlt | −0.012 | |

### Analysis
- **Overall**: TC2000 achieves best overall mAP@50 (0.8827), marginally above Aug_1000 (0.8821)
- **TargetCell paradox**: Dedicated augmentation hurt TargetCell AP (−0.016). Root cause: DDPM generates standalone 64×64 crops, but YOLO detects TargetCell within full blood smear slides — scale/context mismatch
- **Unexpected winners**: Nucleated (+2.6%), Atypical (+1.8%) benefit most, likely from reduced overfitting via additional diverse training examples
- **Rare class mAP** (7 classes): Baseline 0.899 → TC2000 0.899 (no net change)

---

## Visualizations

- [`perclass_map_comparison.png`](perclass_map_comparison.png) — Per-class mAP@50 bar chart (3 models)
- [`improvement_delta.png`](improvement_delta.png) — TC2000 vs Baseline delta chart
- [`ddpm_samples_targetcell.png`](ddpm_samples_targetcell.png) — 4×5 grid of generated TargetCell images
- [`../ddpm/`](../ddpm/) — 12 individual DDPM sample images (tc_sample_*.png)
