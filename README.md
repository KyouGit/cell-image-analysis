# Blood Cell Image Analysis

**YOLO Object Detection + AutoEncoder Latent Space Analysis for Blood Cell Images**

혈구 이미지에 대한 객체 탐지(YOLO) 및 비지도 표현 학습(AutoEncoder + UMAP/t-SNE) 파이프라인.
U2Bio에서의 실무 경험을 바탕으로, 공개 데이터셋(BCCD)을 활용하여 재구현한 포트폴리오 프로젝트입니다.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Demo Pipeline (BCCD)                      │
│                                                             │
│  BCCD Dataset ──→ YOLOv8 Detection ──→ Cell Crop ──→ AutoEncoder ──→ UMAP/t-SNE │
│  (364 images)     (mAP@50: 0.913)     (6,850 cells)  (latent=256)    Visualization │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Production System (U2Bio)                       │
│                                                             │
│  Client ──→ Flask API ──→ YOLOv10 Detection ──→ VAE OOD Filter ──→ Response │
│             (HTTPS/JWT)   (25-class, 318K imgs)  (blur+anomaly)    MariaDB Log │
└─────────────────────────────────────────────────────────────┘
```

## Results

### YOLO Detection (YOLOv8n, BCCD Dataset)

| Metric | All | WBC | RBC | Platelet |
|--------|-----|-----|-----|----------|
| **mAP@50** | **0.913** | 0.976 | 0.853 | 0.909 |
| Precision | 0.920 | 0.975 | 0.854 | 0.931 |
| Recall | 0.868 | 0.945 | 0.780 | 0.879 |

<p align="center">
  <img src="assets/detection_example.jpg" width="45%" alt="Detection Example"/>
  <img src="assets/confusion_matrix.png" width="45%" alt="Confusion Matrix"/>
</p>
<p align="center">
  <img src="assets/pr_curve.png" width="60%" alt="PR Curve"/>
</p>

### AutoEncoder Latent Space Analysis

| Metric | Value |
|--------|-------|
| Val Loss (MSE) | 0.0023 |
| Latent Dim | 256 |
| Cropped Cells | 6,850 (WBC: 387, RBC: 5,993, Platelet: 470) |

<p align="center">
  <img src="assets/umap_visualization.png" width="45%" alt="UMAP"/>
  <img src="assets/tsne_visualization.png" width="45%" alt="t-SNE"/>
</p>
<p align="center">
  <img src="assets/reconstruction.png" width="60%" alt="Reconstruction"/>
</p>
<p align="center">
  <img src="assets/class_overlap_analysis.png" width="50%" alt="Class Overlap"/>
  <img src="assets/training_curves.png" width="45%" alt="Training Curves"/>
</p>

## Quick Start

```bash
git clone https://github.com/KyouGit/cell-image-analysis.git
cd cell-image-analysis
pip install -r requirements.txt

# Step 1: Download BCCD dataset + VOC→YOLO format conversion
python pipeline/prepare_data.py

# Step 2: YOLO detection training + cell cropping
python pipeline/yolo_detect.py --mode all

# Step 3: AutoEncoder training + UMAP/t-SNE visualization
python pipeline/main.py
```

## Project Structure

```
cell-image-analysis/
├── pipeline/                    # Reproducible demo pipeline
│   ├── prepare_data.py          #   BCCD download + VOC→YOLO conversion
│   ├── yolo_detect.py           #   YOLOv8 train/eval/crop
│   ├── main.py                  #   AutoEncoder + UMAP/t-SNE
│   └── example.py               #   Detection demo & similar cell search
│
├── production_reference/        # Sanitized production code (U2Bio)
│   ├── app.py                   #   Flask REST API (HTTPS, JWT, batch inference)
│   ├── u2_utils.py              #   YOLOv10 inference, VAE OOD, DB logging
│   ├── u2_imagesplitter.py      #   High-res image tiling + annotation remap
│   ├── README.md                #   Production system documentation
│   └── notebooks/               #   Analysis notebooks
│       ├── umap_platelet.ipynb  #     PLT subtype UMAP (23K images)
│       ├── umap_rbc.ipynb       #     RBC morphology UMAP (318K images)
│       ├── umap_wbc.ipynb       #     WBC subtype analysis
│       ├── umap_presentation.ipynb  # Presentation-ready visualizations
│       ├── yolo_cam.ipynb       #     Class Activation Maps
│       └── yolo_monitoring.ipynb #    Training monitoring (Dual A100)
│
├── assets/                      # Result images for README
├── requirements.txt
└── .gitignore
```

## Demo vs Production

| | Demo (This Repo) | Production (U2Bio) |
|--|---|---|
| **Detection** | YOLOv8n (3 classes) | YOLOv10s (25 classes) |
| **Dataset** | BCCD 364 images | 318,000+ annotated cells |
| **Input** | 640×640 | 384×384 |
| **OOD Filter** | None | VAE reconstruction + Laplacian blur |
| **Serving** | Local script | Flask HTTPS + JWT + MariaDB |
| **GPU** | Single GPU | Dual NVIDIA A100 |
| **Inference** | Sequential | ThreadPoolExecutor (8 workers) |

## Tech Stack

- **Detection**: Ultralytics YOLOv8 / YOLOv10
- **Representation Learning**: PyTorch (ConvAutoEncoder, VAE)
- **Visualization**: UMAP, t-SNE, matplotlib, seaborn
- **Production**: Flask, PyMySQL, JWT, OpenCV
- **Training**: CosineAnnealingLR, AdamW

## Dataset

This demo uses the [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) (Blood Cell Count and Detection).
- 364 blood smear images with bounding box annotations
- 3 classes: WBC, RBC, Platelet
- License: MIT

## Contact

- GitHub: [@KyouGit](https://github.com/KyouGit)
- Email: qsc303@gmail.com

## License

MIT License
