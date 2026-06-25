# Experiments — PBS 혈구 AI 연구 실험 전체 정리

말초혈액도말(PBS) 25-class 혈구 탐지 시스템을 구축하면서 수행한 실험들.  
아주대학교병원 IRB 승인 데이터 (진단검사의학과 전문의 어노테이션, 318,000+ bounding box).

> **서버 없이 결과 확인하는 방법**
> ```bash
> # 결과 JSON → 그래프 출력 (모델·GPU 불필요)
> python experiments/results/visualize.py
> python experiments/results/visualize.py --save   # PNG 저장
> ```

---

## 전체 실험 흐름

```
[문제 인식] Precision 0.68 — 혼동쌍 label noise 심각
     │
     ├─ EXP-001  YOLOv5 → v8 → v10 → v11s 비교 실험
     │           └─ 결론: YOLOv11s 채택 (mAP@50 0.879)
     │
     ├─ EXP-003  Label Noise 처리
     │           └─ 결론: UMAP 경계 탐지 + 전문의 재라벨링 → Precision 0.90
     │
     ├─ EXP-004  Deep Ensemble + WBF
     │           └─ 결론: 재현성 확보 + FP 감소
     │
     ├─ EXP-005  캘리브레이션 (Temperature Scaling / Conformal Prediction)
     │           └─ 결론: 불확실성 정량화, 임상 신뢰도 향상
     │
     ├─ EXP-005c XAI 분석 (EigenCAM v1→v7 / D-RISE)
     │           └─ 결론: 오분류 원인 규명, 어노테이션 가이드라인 개선
     │
     ├─ EXP-L2   SupCon 표현 학습
     │           └─ 결론: 혼동쌍 분리 44.5×, Linear Probe 91.2%
     │
     ├─ EXP-G    DDPM 희소 클래스 증강 → 재학습
     │           └─ 결론: 전체 +0.27%, 희소 클래스 선택적 개선
     │
     ├─ EXP-VAE  VAE OOD 필터
     │           └─ 결론: 블러·이물질 자동 제거
     │
     └─ EXP-TS   Two-Stage 재분류 실험 (negative)
                 └─ 결론: SupCon은 탐지 대체 불가, 표현 분석 도구로만 유효
```

---

## EXP-001 · YOLO 모델 비교

**목적:** 25-class 혈구 탐지에 최적인 YOLO 버전 선택

| 모델 | mAP@50 | 특이사항 |
|------|--------|---------|
| YOLOv5 | — | 3-class 탐지, label noise로 Precision 0.68 |
| YOLOv8 | — | 25-class로 확장, FastAPI 배포 시작 |
| YOLOv10 | — | anchor-free, 성능 유사 |
| **YOLOv11s** | **0.879** | 최종 채택 — 속도·정확도 균형, 소형 세포 강점 |

**코드:**
- `training/run_yolo_train.py` — 기본 학습
- `archive/api_v1/app.py` — YOLOv5 Flask 배포 (초기)
- `archive/api_v2/` — YOLOv8 FastAPI 배포

---

## EXP-003 · Label Noise 처리

**문제:** 형태학적으로 유사한 혈구 간 어노테이터 불일치 (inter-rater variability)

```
혼동쌍 예시
  WBC: BandNeutrophil ↔ SegNeutrophil   (핵 분절 정도 차이)
       Myelocyte      ↔ Blast            (미성숙 세포 구분)
  RBC: TargetCell     ↔ RBC              (중심창백 여부)
       Schistocyte    ↔ RBC              (파편 여부)
  PLT: GiantPlt       ↔ PLT             (크기 기준 모호)
```

**접근법:**
1. UMAP 잠재 공간에서 혼동쌍 경계 샘플 자동 탐지
2. 경계 샘플 → 진단검사 전문의 재라벨링
3. `label_smoothing=0.1` 적용 (YOLO 학습 파라미터)

**결과:**

| | 이전 | 이후 |
|--|------|------|
| Precision | 0.68 | **0.90** |
| 개선폭 | | **+32%** |

---

## EXP-004 · Deep Ensemble + WBF

**목적:** 단일 모델의 불안정성 해소, 불확실성 추정

- 동일 데이터·코드로 seed 0~4 각각 독립 학습 (5개 모델)
- 5개 모델 예측을 **WBF(Weighted Box Fusion)**으로 앙상블
- per-detection 불확실성: 5개 모델 confidence 분산으로 추정

**코드:** `training/ensemble_train.py`

```python
# ensemble_train.py 핵심 구조
for seed in range(5):
    set_seed(seed)
    model = YOLO('yolov11s.pt')
    model.train(data='pbs06.yaml', epochs=100, seed=seed,
                project=f'EXP-006_DeepEnsemble/seed_{seed}')
```

---

## EXP-005 · 캘리브레이션

**목적:** 모델 confidence가 실제 정확도를 얼마나 잘 반영하는지 측정·개선

### Temperature Scaling
- validation set으로 temperature T 최적화
- Reliability Diagram (10-bin)으로 캘리브레이션 전·후 비교
- ECE(Expected Calibration Error) 감소 확인

### Selective Prediction
- confidence threshold별 Risk-Coverage Curve
- 예: threshold 0.8 설정 시 낮은 confidence 탐지 기각 → 정확도 상승

### Conformal Prediction
- α=0.10 (τ=0.475) 기준: 90% 확률로 true class 포함 보장
- 임상에서 "모델이 불확실한 경우 의료진에게 알림" 용도

---

## EXP-005c · XAI — EigenCAM (v1 → v7)

**목적:** YOLO가 혼동쌍을 오분류할 때 어떤 영역을 보는지 시각화

| 버전 | 내용 |
|------|------|
| v1 | FPN P3/P4/P5 heatmap 기본 생성 |
| v4 | 혼동쌍 8종 한 번에 처리 |
| v6 | 전체 25-class 배치 처리 |
| **v7** | **논문 제출본** — WBC×4 / RBC×4 / PLT×2 혼동쌍, D-RISE 정량 비교 |

**코드:** `eigencam/run_eigencam_v1.py` ~ `v7.py`

**주요 발견:**
```
BandNeutrophil vs SegNeutrophil 오분류 시
→ 모델이 핵 분절 패턴(진짜 차이)이 아닌 세포 크기에 집중
→ 어노테이션 가이드라인에 크기 기준 명시 추가
```

---

## EXP-L2 · SupCon 표현 학습

**목적:** YOLO FPN feature vs SupCon feature — 혼동쌍 분리 능력 비교

**모델:** ResNet50 + Supervised Contrastive Loss

```
학습: val set features로 Linear Probe 학습
      → Logistic Regression (25-class)
평가: Linear Probe Accuracy, Silhouette Score, 혼동쌍 cosine distance
```

**결과:**

| 지표 | YOLO FPN | SupCon |
|------|----------|--------|
| Linear Probe Accuracy | — | **0.912** |
| Silhouette Score | 0.112 | **0.500** |
| 혼동쌍 cosine distance | 1.0× (baseline) | **44.5×** 향상 |

**코드:** `training/supcon_train.ipynb`

**해석:** SupCon feature space에서 혼동쌍이 명확히 분리됨.
이를 근거로 "YOLO FPN보다 SupCon이 세포 표현에 더 적합"을 논문에서 주장.

---

## EXP-G · DDPM 희소 클래스 증강

**문제:** 희소 클래스(BandNeutrophil 153개, hyperSeg 130개 등) 데이터 부족

**방법:** DDPM(Diffusion Model)으로 합성 이미지 생성

**생성 품질 (FID — 낮을수록 실제와 유사):**

| 클래스 | FID |
|--------|-----|
| hyperSeg | 58.6 |
| BandNeutrophil | 66.6 |
| Stomatocyte | 74.2 |
| Schistocyte | 87.7 |
| TargetCell | 96.6 |
| **평균** | **76.7** |

**재학습 결과** (`results/ddpm_augmentation.json`):

| 조건 | mAP@50 | 전체 변화 |
|------|--------|---------|
| Baseline | 0.8794 | — |
| Aug_500 (합성 +500/class) | 0.8811 | +0.17% |
| **Aug_1000 (합성 +1000/class)** | **0.8821** | **+0.27%** |

**클래스별 개선 (Aug_1000):**

| 클래스 | Baseline | Aug_1000 | Δ |
|--------|----------|----------|---|
| Atypical | 0.759 | 0.787 | **+2.8%** |
| Nucleated | 0.868 | 0.884 | **+1.6%** |
| TargetCell | 0.659 | 0.665 | +0.6% |

**결론:** 전체 mAP 소폭 향상. 원래 성능이 낮았던 희소·난이도 높은 클래스에서 선택적으로 유효.  
**코드:** `training/ddpm_train.ipynb`, `training/run_yolo_train2.py`

---

## EXP-VAE · VAE 기반 OOD 필터

**목적:** 블러·이물질·비혈구 이미지를 추론 전에 자동 필터링

**아키텍처:** `ResidualBlock → VAE (z_dim=128) → UNet decoder`

```
학습: PBS 혈구 이미지로 정상 분포 학습
추론: reconstruction error 계산
      → mean + 3σ 초과 시 OOD 판정 → 탐지 스킵
검증: food101 / CIFAR-10 / ImageNet으로 크로스 도메인 테스트
```

**코드:** `training/vae_ood.py`

```bash
# 학습
python experiments/training/vae_ood.py --train

# 단일 이미지 OOD 판정
python experiments/training/vae_ood.py --eval data/sample/BloodImage_00001.jpg
```

---

## EXP-TS · Two-Stage 재분류 (Negative Result)

**가설:** SupCon이 혼동쌍을 잘 분리하니, YOLO 탐지 후 SupCon으로 재분류하면 AP가 오를 것이다.

**방법:**
```
YOLO 탐지 → bounding box crop → SupCon feature 추출 → Linear Classifier 예측
                                 (SUPCON_BATCH=256 배치 처리)
```

**결과** (`results/twostage_reclassification.json`):

| | YOLO 단독 | Two-Stage |
|--|-----------|-----------|
| **전체 mAP@50** | **0.779** | **0.041** |
| RBC | 0.951 | 0.539 |
| 나머지 23개 클래스 | 0.28~0.99 | ≈ 0.0 |

**실패 원인:**
1. YOLO가 검출한 crop은 경계가 부정확하거나 작은 세포가 섞임 → SupCon 혼란
2. 클래스 불균형 (RBC 63,579개 vs BandNeutrophil 153개) → 분류기가 RBC로 몰아 예측
3. SupCon linear probe val accuracy 91.2%는 **깨끗한 crop 기준** — 실전 crop과 다름

**결론 (논문 기여):**
> SupCon은 표현 분석 도구(UMAP, linear probe)로 유효하지만,  
> YOLO의 공간 컨텍스트를 활용한 end-to-end 탐지를 대체할 수 없다.  
> → Paper 2/3의 "SupCon은 표현 분석 도구" 주장을 실험으로 뒷받침.

**코드:** `training/run_twostage.py`

---

## 결과 파일 목록

| 파일 | 내용 |
|------|------|
| `results/ddpm_augmentation.json` | DDPM 3조건 재학습 결과 (per-class AP@50) |
| `results/twostage_reclassification.json` | Two-Stage 결과 (25 class + overall) |
| `results/visualize.py` | 위 JSON → 그래프 (서버·GPU 불필요) |

```bash
# 결과 시각화 (JSON만 있으면 됨)
python experiments/results/visualize.py

# PNG로 저장
python experiments/results/visualize.py --save
# → experiments/results/figures/ 에 저장됨
```

---

## 폴더 구조

```
experiments/
├── eigencam/                      # XAI 실험 (EXP-005c)
│   ├── run_eigencam_v1.py         #   초기 heatmap
│   ├── run_eigencam_v4.py         #   혼동쌍 8종
│   ├── run_eigencam_v6.py         #   25-class 배치
│   ├── run_eigencam_v7.py         #   논문 제출본 ★
│   ├── run_schisto_cam_candidates.py
│   └── run_schistocyte_only.py
│
├── training/                      # 학습 실험
│   ├── ensemble_train.py          #   Deep Ensemble 5 seeds (EXP-004)
│   ├── run_yolo_train.py          #   YOLOv11s 기본 (EXP-001)
│   ├── run_yolo_train2.py         #   DDPM 증강 재학습 (EXP-G4)
│   ├── run_twostage.py            #   Two-Stage 재분류 (EXP-TS) ★
│   ├── supcon_train.ipynb         #   SupCon ResNet50 (EXP-L2)
│   ├── ddpm_train.ipynb           #   DDPM 생성 모델 (EXP-G1)
│   └── vae_ood.py                 #   VAE OOD 필터 (EXP-VAE) ★
│
├── results/                       # 실험 결과 (서버 없이 확인 가능)
│   ├── ddpm_augmentation.json     #   DDPM 재학습 결과 ★
│   ├── twostage_reclassification.json  # Two-Stage 결과 ★
│   ├── visualize.py               #   결과 시각화 스크립트 ★
│   └── README.md
│
└── archive/                       # 배포 버전 이력
    ├── pipeline_v1.py             #   초기 PBS 파이프라인 (YOLOv5)
    ├── api_v1/app.py              #   Flask API (YOLOv5)
    ├── api_v2/                    #   FastAPI (YOLOv8)
    └── api_dev/u2_utils.py        #   SupCon 재분류 개발 버전
```
