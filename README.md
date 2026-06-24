# Blood Cell Image Analysis

**Peripheral Blood Smear (PBS) AI — End-to-end research from label noise handling to production deployment**

말초혈액도말(PBS) 이미지의 25-class 혈구 탐지 및 표현 학습 시스템.  
아주대학교병원 IRB 승인 데이터(진단검사 전문의 어노테이션, 318K+ 이미지)를 기반으로,  
label noise 처리 → 모델 개선 → XAI 분석 → 제품 배포까지 end-to-end로 수행했습니다.

> U2Bio 실무 경험 기반 포트폴리오. 민감 데이터는 제거하고 구조와 방법론만 공개.

---

## Research Challenges & Solutions

### 1. Label Noise — Precision 0.68 → 0.90

임상 데이터의 고유한 문제인 inter-rater variability(형태학적으로 유사한 혈구 간 어노테이터 불일치)를 해결하기 위해 다단계 접근법을 적용했습니다.

| 단계 | 문제 | 접근법 | 결과 |
|------|------|--------|------|
| 초기 모델 | 혼동쌍 클래스 간 경계 모호 | YOLOv5 baseline | Precision 0.68 |
| 모델 개선 | 아키텍처 한계 | YOLOv8 → YOLOv10 → **YOLOv11s** 비교 실험 | mAP@50 0.879 |
| Label noise 처리 | Hard case 경계 샘플 오라벨링 | UMAP 잠재 공간에서 경계 샘플 탐지 → 진단검사 전문의 재라벨링 | — |
| Soft-label 전략 | 형태학적으로 유사한 클래스 | 혼동쌍(BandNeutrophil↔SegNeutrophil 등) soft-label 재정의 | **Precision 0.90** |
| OOD 필터링 | 블러·이물질 이미지 혼입 | VAE reconstruction error + Laplacian blur 검출 | FP 감소 |

**혼동쌍 (주요 hard case):**
- WBC: BandNeutrophil ↔ SegNeutrophil, Myelocyte ↔ Blast, Monocyte ↔ Lymphocyte
- RBC: TargetCell ↔ RBC, Echinocyte ↔ RBC, Schistocyte ↔ RBC
- PLT: GiantPlt ↔ PLT, clump-PLT ↔ PLT

---

### 2. Representation Analysis — SupCon vs YOLO FPN

YOLO FPN feature가 아닌 Supervised Contrastive Learning(SupCon) 기반 표현이 혼동쌍을 얼마나 잘 분리하는지 정량 분석했습니다.

| 지표 | YOLO FPN | SupCon (ResNet50) |
|------|----------|-------------------|
| Linear Probe Accuracy | — | **0.912** |
| Silhouette Score | 0.112 | **0.500** |
| 혼동쌍 cosine distance | 1.0× (baseline) | **44.5×** |

<p align="center">
  <img src="assets/representation/umap_yolo_fpn.png" width="45%" alt="YOLO FPN UMAP"/>
  <img src="assets/representation/umap_supcon.png" width="45%" alt="SupCon UMAP"/>
</p>
<p align="center">
  <img src="assets/representation/confusion_pair_distance.png" width="55%" alt="Confusion Pair Distance"/>
  <img src="assets/representation/linear_probe_confusion_matrix.png" width="40%" alt="Linear Probe CM"/>
</p>

---

### 3. XAI — EigenCAM & D-RISE 기반 오분류 분석

모델이 혼동쌍을 오분류할 때 어떤 영역에 집중하는지 분석하여, 어노테이션 개선 방향을 도출했습니다.

```
혼동쌍 이미지 → EigenCAM (FPN layer별 활성화) → saliency divergence 정량화
                → D-RISE (perturbation 기반) → 오분류 케이스 분석
```

- **EigenCAM**: 세포 형태학적 특징 집중 여부 시각화 (핵 형태, 과립 패턴)
- **D-RISE**: 탐지 결과에 대한 pixel-level 중요도 맵 생성
- 결과: BandNeutrophil ↔ SegNeutrophil 혼동 시 모델이 핵 분절 패턴보다 세포 크기에 집중함을 발견 → 어노테이션 가이드라인 보완

<p align="center">
  <img src="assets/xai/eigencam_BandNeutrophil_vs_SegNeutrophil.png" width="45%" alt="EigenCAM Band vs Seg"/>
  <img src="assets/xai/drise_BandNeutrophil_vs_SegNeutrophil.png" width="45%" alt="D-RISE Band vs Seg"/>
</p>
<p align="center">
  <img src="assets/xai/eigencam_Myelocyte_vs_Blast.png" width="30%" alt="EigenCAM Myelocyte vs Blast"/>
  <img src="assets/xai/eigencam_TargetCell_vs_RBC.png" width="30%" alt="EigenCAM TargetCell vs RBC"/>
  <img src="assets/xai/eigencam_Schistocyte_vs_RBC.png" width="30%" alt="EigenCAM Schistocyte vs RBC"/>
</p>

---

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
│  Client ──→ FastAPI ──→ YOLOv11s Detection ──→ SupCon Reclassifier ──→ Response │
│             (Docker/EC2)  (25-class, 318K+ imgs)  (hard case 재분류)    MariaDB Log │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

### Production Performance (U2Bio, 25-class)

<p align="center">
  <img src="assets/results/wbf_class_gain.png" width="55%" alt="WBF Ensemble Class Gain"/>
  <img src="assets/results/reliability_diagram.png" width="40%" alt="Reliability Diagram"/>
</p>
<p align="center">
  <img src="assets/results/risk_coverage_curve.png" width="50%" alt="Risk-Coverage Curve"/>
</p>

### YOLO Detection (YOLOv8n, BCCD Dataset — Demo)

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

---

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

---

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
│   ├── app.py                   #   FastAPI REST API (Docker/EC2 배포)
│   ├── u2_utils.py              #   YOLOv11 inference, VAE OOD, DB logging
│   ├── u2_imagesplitter.py      #   High-res image tiling + annotation remap
│   ├── README.md                #   Production system documentation
│   └── notebooks/               #   Analysis notebooks
│       ├── umap_platelet.ipynb  #     PLT subtype UMAP (23K images)
│       ├── umap_rbc.ipynb       #     RBC morphology UMAP (318K images)
│       ├── umap_wbc.ipynb       #     WBC subtype analysis
│       ├── umap_presentation.ipynb  # Presentation-ready visualizations
│       ├── yolo_cam.ipynb       #     EigenCAM / D-RISE XAI analysis
│       └── yolo_monitoring.ipynb #    Training monitoring (Dual A100)
│
├── assets/                      # Result images for README
├── requirements.txt
└── .gitignore
```

---

## Demo vs Production

| | Demo (This Repo) | Production (U2Bio) |
|--|---|---|
| **Detection** | YOLOv8n (3 classes) | YOLOv11s (25 classes) |
| **Dataset** | BCCD 364 images | 318,000+ annotated cells (아주대 IRB) |
| **Annotators** | Public label | 진단검사 전문의 |
| **Label Strategy** | Hard label | Soft-label (혼동쌍 재정의) |
| **OOD Filter** | None | VAE reconstruction + Laplacian blur |
| **XAI** | None | EigenCAM + D-RISE |
| **Serving** | Local script | FastAPI + Docker + AWS EC2 |
| **GPU** | Single GPU | Dual NVIDIA A100 |
| **Inference** | Sequential | ThreadPoolExecutor (8 workers) |

---

## Tech Stack

- **Detection**: Ultralytics YOLOv8 / YOLOv11, Deep Ensemble (5 seeds), WBF
- **Representation Learning**: PyTorch (SupConResNet50, ConvAutoEncoder, β-VAE)
- **Label Noise**: Soft-label strategy, UMAP-based boundary sample detection
- **XAI**: EigenCAM (FPN layer), D-RISE (perturbation-based saliency)
- **Visualization**: UMAP, t-SNE, matplotlib, seaborn
- **Production**: FastAPI, uvicorn, Docker, AWS EC2, MariaDB, JWT
- **Export**: ONNX (YOLOv11s best.onnx)
- **Training**: CosineAnnealingLR, AdamW, W&B experiment tracking

---

## Dataset

**Demo:** [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) — 364 blood smear images, 3 classes, MIT License

**Production (U2Bio):**
- 아주대학교병원 IRB 승인 말초혈액도말 이미지
- 진단검사의학과 전문의 어노테이션
- 318,000+ bounding box, 25 classes
- 데이터 비공개 (회사 소유) — available upon reasonable request

---

## Contact

- GitHub: [@KyouGit](https://github.com/KyouGit)
- Email: qsc303@gmail.com

## License

MIT License
