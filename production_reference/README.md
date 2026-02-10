# Production Reference (U2Bio)

> **Note:** This directory contains sanitized versions of production code used at U2Bio.
> Company data, model weights, server paths, and credentials have been removed.

## System Overview

U2Bio의 혈구 분석 AI 시스템에서 실제 사용된 코드입니다.

### Architecture

```
Client (U2Labeler) → Flask API Server → YOLOv10 Detection → VAE OOD Filter → Response
                                      ↓
                               MariaDB Logging
```

### Components

| File | Role |
|------|------|
| `app.py` | Flask REST API (HTTPS, JWT auth, multi-threaded batch inference) |
| `u2_utils.py` | YOLOv10 inference, VAE OOD detection, DB logging, edge detection |
| `u2_imagesplitter.py` | High-resolution image tiling with annotation remapping |

### Key Features
- **25-class cell detection** (WBC/RBC/Platelet variants + abnormal morphologies)
- **VAE-based OOD filtering** (reject blurry or non-blood-cell images)
- **Batch inference** with ThreadPoolExecutor (8 workers)
- **JWT authentication** + MariaDB audit logging

### Notebooks

| Notebook | Description |
|----------|-------------|
| `umap_platelet.ipynb` | AutoEncoder + UMAP visualization for PLT subtypes (23K images) |
| `umap_rbc.ipynb` | UMAP clustering of 12 RBC morphology types (318K images) |
| `umap_wbc.ipynb` | WBC subtype analysis (Neutrophil, Lymphocyte, Blast, etc.) |
| `umap_presentation.ipynb` | Consolidated presentation-ready UMAP visualizations |
| `yolo_cam.ipynb` | Class Activation Maps for YOLOv10 interpretability |
| `yolo_monitoring.ipynb` | Training monitoring (Dual A100, 800 epochs, AdamW) |

### Production Specs
- **GPU**: Dual NVIDIA A100 (19.7GB VRAM)
- **Dataset**: 318,000+ annotated cell images
- **Model**: YOLOv10s (384x384 input)
- **Server**: Linux, Flask + HTTPS, MariaDB
