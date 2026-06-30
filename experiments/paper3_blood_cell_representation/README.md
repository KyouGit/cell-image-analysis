# Paper 3 — Blood Cell Representation Analysis

Analysis code and results for the Expert Systems (Wiley) submission:

**"Comparative Analysis of Representation Methods for Blood Cell Morphology in DI-60 PBS Images"**

## Dataset

DI-60 PBS dataset: 31,536 images, 25 morphological classes, patient-level split.

- Train: 18,958 / Val: 6,298 / Test: 6,280 images

Raw images are not publicly distributed due to patient privacy constraints.  
Contact the corresponding author (u240602@u2bio.com) for data access subject to IRB approval.

## Scripts

| File | Description |
|------|-------------|
| `scripts/exp_l3_supcon_train.py` | Standard SupCon (ResNet-50) training — EXP-L3 |
| `scripts/exp_l3b_supcon_hn.py` | SupCon with hard negative weighting — EXP-L3b (ablation) |
| `scripts/exp_p3_server.py` | Cross-model comparison analysis (YOLO FPN, DINOv2, β-VAE, SupCon) |

## Results

| File | Description |
|------|-------------|
| `results/all_pair_distances.csv` | Cosine distances for all 300 class-pair combinations across methods |
| `results/per_class_compactness.csv` | Per-class intra-cluster compactness and AP50 for correlation analysis |

## Key Findings

- **EXP-L3b ablation**: Hard negative weighting provides no measurable improvement over standard SupCon (kNN-5: 0.906, Silhouette: 0.506, ARI: 0.250, LP: 0.888)
- **YOLO FPN** achieves the smallest confusion-pair distances (mean 0.00097), reflecting task-aligned training
- **DINOv2** provides the best generalist representation (kNN-5: 0.956)
- **SupCon** provides the best class separation (Silhouette: 0.506)

## Requirements

```
torch>=2.0
torchvision
scikit-learn
umap-learn
numpy
pandas
ultralytics
```
