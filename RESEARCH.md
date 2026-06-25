# 연구 요약 핸드오프 파일
> 작성일: 2026-06-25  
> 작성자: 최규민 (유투바이오 재직 중)  
> 용도: 경력기술서·포트폴리오 작성용 연구 정보 전달

---

## 1. 연구 개요

**소속:** 유투바이오 (U2Bio)  
**프로젝트:** 말초혈액도말(PBS, Peripheral Blood Smear) 혈구 AI 탐지 시스템  
**데이터:** 아주대학교병원 IRB 승인, 진단검사의학과 전문의 어노테이션, 318,000+ bounding box, 25 classes  
**목표:** 임상 현장 적용 가능한 실시간 혈구 탐지·분류 AI 시스템 연구 및 배포

---

## 2. 핵심 기술 문제와 해결 접근법

### 2-1. Label Noise & Inter-rater Inconsistency
**문제:** 형태학적으로 유사한 혈구(혼동쌍) 간 어노테이터 불일치 → Precision 0.68 수준

**접근법:**
1. UMAP 잠재 공간에서 혼동쌍 경계 샘플 자동 탐지
2. 탐지된 경계 샘플을 진단검사 전문의가 재라벨링
3. YOLO `label_smoothing=0.1` + soft-label 전략 적용

**결과:** Precision 0.68 → **0.90** (개선 +32%)

**주요 혼동쌍:**
- WBC: BandNeutrophil ↔ SegNeutrophil, Myelocyte ↔ Blast, Monocyte ↔ Lymphocyte
- RBC: TargetCell ↔ RBC, Echinocyte ↔ RBC, Schistocyte ↔ RBC
- PLT: GiantPlt ↔ PLT, clump-PLT ↔ PLT

---

### 2-2. 탐지 모델 개발 (YOLOv11s)
**모델 비교 실험:** YOLOv5 → YOLOv8 → YOLOv10 → YOLOv11s (순차 비교)

**최종 모델 성능 (YOLOv11s, test set):**
- mAP@50: **0.879**
- Precision: **0.90**
- 학습 데이터: 25 classes, 318K+ images (아주대 IRB)
- 추론 환경: Dual NVIDIA A100

**WBF(Weighted Box Fusion) 앙상블:**
- Deep Ensemble 5 seeds → WBF 적용
- 전체 mAP@50 추가 개선

---

### 2-3. 모델 불확실성 정량화 및 캘리브레이션
**적용 기법:**
1. **Deep Ensemble (5 seeds):** 재현성 확인, per-detection 불확실성 추정
2. **Temperature Scaling:** Calibration 개선 (Reliability Diagram으로 검증)
3. **Selective Prediction:** Risk-Coverage Curve 분석
4. **Conformal Prediction:** α=0.10(τ=0.475) 기준 class별 커버리지 보장

**의의:** 임상 환경에서 모델이 틀릴 가능성을 정량적으로 제공 → 의료진 신뢰도 향상

---

### 2-4. Supervised Contrastive Learning (SupCon) 표현 분석
**목적:** YOLO FPN feature vs SupCon 표현의 혼동쌍 분리 능력 비교

**모델:** ResNet50 기반 SupCon (Supervised Contrastive Loss)

**결과:**
| 지표 | YOLO FPN | SupCon (ResNet50) |
|------|----------|-------------------|
| Linear Probe Accuracy | — | **0.912** (n=2,500 val) |
| Silhouette Score | 0.112 | **0.500** |
| 혼동쌍 cosine distance | 1.0× (baseline) | **44.5×** |

**추가 실험 — Two-Stage 재분류 (negative result):**
- YOLO 탐지 후 SupCon으로 class 재분류 시도
- 결과: mAP@50 0.7791 → 0.0407 (대폭 하락)
- 원인: YOLO detected crop의 품질 불균일 + 클래스 불균형(RBC 과다 예측)
- 결론: SupCon은 탐지 대체가 아닌 표현 분석 도구로 적합 (논문 주장과 일치)

---

### 2-5. DDPM 기반 희소 클래스 데이터 증강
**문제:** 희소 클래스(BandNeutrophil, hyperSeg, Schistocyte, TargetCell, Stomatocyte) 데이터 부족

**접근법:** DDPM(Denoising Diffusion Probabilistic Model)로 합성 이미지 생성

**생성 품질 (FID):**
| 클래스 | FID |
|--------|-----|
| BandNeutrophil | 66.60 |
| hyperSeg | 58.61 |
| Schistocyte | 87.73 |
| TargetCell | 96.60 |
| Stomatocyte | 74.17 |
| **평균** | **76.74** |

**재학습 결과 (3조건 비교):**
| 조건 | mAP@50 | 증감 |
|------|--------|------|
| Baseline | 0.8794 | — |
| Aug_500 (합성 +500/class) | 0.8811 | +0.0017 |
| Aug_1000 (합성 +1000/class) | 0.8821 | **+0.0027** |

**클래스별 주요 개선:**
| 클래스 | Baseline | Aug_1000 | 변화 |
|--------|----------|----------|------|
| TargetCell | 0.6590 | 0.6654 | +0.0064 |
| Nucleated | 0.8680 | 0.8837 | **+0.0157** |
| Atypical | 0.7594 | 0.7874 | **+0.0280** |

**해석:** 전체 mAP 소폭 향상, 성능 낮았던 희소·난이도 높은 클래스에서 선택적으로 유효

---

### 2-6. XAI (설명가능 AI) 분석
**적용 기법:**
1. **EigenCAM:** YOLOv11 FPN layer별 활성화 맵 시각화 (11개 혼동쌍)
2. **D-RISE:** Perturbation 기반 pixel-level 중요도 맵 (탐지 결과 단위)

**주요 발견:**
- BandNeutrophil ↔ SegNeutrophil 혼동 시 모델이 핵 분절 패턴보다 세포 크기에 집중
- D-RISE: YOLO FPN 특성상 diffuse saliency → EigenCAM이 형태 집중에 더 적합
- 분석 결과로 어노테이션 가이드라인 보완

---

### 2-7. VAE 기반 OOD 필터
**목적:** 블러·이물질·비혈구 이미지 자동 필터링

**아키텍처:** ResidualBlock + VAE + UNet skip connections (`AdvancedVAE_UNet`, z_dim=128)

**방법:** PBS 혈구 이미지로 학습 후, reconstruction error 3σ threshold 초과 시 OOD 판정

**검증:** food101, CIFAR-10, ImageNet 크로스 도메인 OOD 검증

**효과:** FP(False Positive) 감소

---

## 3. 프로덕션 배포

**스택:** FastAPI + Docker(u2python base image) + AWS EC2

**파이프라인:**
```
Client 이미지 전송 (base64)
→ FastAPI /inf 엔드포인트 (포트 8000)
→ YOLOv11s 탐지 (25 class)
→ SupCon 재분류 (hard case)
→ MariaDB 로그 저장
→ JSON 결과 반환
```

**최적화:**
- ONNX 변환 (YOLOv11s best.onnx)
- ThreadPoolExecutor 8 workers 병렬 추론
- 클래스별 confidence threshold 차등 적용 (RBC계열 0.6)

**API 형식:**
```json
POST /inf
{
  "model_name": "yolov11s_pbs",
  "width": 640, "height": 480,
  "image": "data:image/jpeg;base64,..."
}
→ {"result_code": "OK", "result": [{"class": "RBC", "x1":..., "prob":...}]}
```

---

## 4. 기술 스택 전체

| 분류 | 기술 |
|------|------|
| 탐지 모델 | YOLOv5 → YOLOv8 → YOLOv10 → YOLOv11s |
| 표현 학습 | SupCon (ResNet50), ConvAutoEncoder, β-VAE |
| 생성 모델 | DDPM |
| OOD 필터 | VAE-UNet (ResidualBlock + VAE + UNet) |
| 앙상블 | Deep Ensemble (5 seeds), WBF |
| 캘리브레이션 | Temperature Scaling, Conformal Prediction |
| XAI | EigenCAM, D-RISE |
| 시각화 | UMAP, t-SNE, matplotlib, seaborn |
| 배포 | FastAPI, Docker, AWS EC2, ONNX |
| DB | MariaDB |
| 학습 인프라 | Dual NVIDIA A100, nohup 백그라운드 |
| 프레임워크 | PyTorch, scikit-learn, Ultralytics |

---

## 5. 논문 현황

| 논문 | 내용 | 상태 | 타겟 |
|------|------|------|------|
| Paper 1 | YOLOv11 탐지 + 캘리브레이션 + XAI 분석 | docx 완성, 투고 대기 | JBER (국내) |
| Paper 2 | 잠재공간 분석 (SupCon vs YOLO FPN) | docx 완성 | 국내 학술지 |
| Paper 3 | 표현 분석 영문 버전 | docx 완성 | 국제 저널 |
| Paper 4 | DDPM 희소 클래스 증강 | docx 완성 (수치 반영 완료) | 국내 학술지 |
| 통합 SCI | 위 1~4 통합 | 기획 중 | Computers in Biology and Medicine (IF 6.3) |

---

## 6. VUNO PXI JD 매핑 (경력기술서 작성 참고)

| JD 요구사항 | 해당 경험 | 강도 |
|-------------|-----------|------|
| Label noise, rater variability | UMAP 경계 샘플 탐지 + 전문의 재라벨링 → Precision +32% | ★★★ |
| Self-/semi-supervised learning | SupCon 표현 학습 (linear probe 91.2%) | ★★★ |
| Data scarcity 해결 | DDPM 희소 클래스 증강 (5 classes, FID~76) | ★★★ |
| 모델 캘리브레이션/불확실성 | Deep Ensemble + Temperature Scaling + Conformal Prediction | ★★★ |
| 제품 배포 경험 | FastAPI + Docker + AWS EC2 실서비스 운영 | ★★★ |
| ONNX 추론 최적화 | YOLOv11s ONNX 변환 + 병렬 추론 | ★★ |
| 임상·규제 협업 | IRB 승인 데이터, 진단검사 전문의 협업, 논문 근거 정리 | ★★★ |
| 재현 가능한 실험 체계 | 3조건(Baseline/Aug_500/Aug_1000) 체계적 비교, 결과 JSON 저장 | ★★ |
| XAI 분석 | EigenCAM + D-RISE로 오분류 원인 규명 | ★★ |
| 의료 AI 도메인 | PBS 혈구 25 class, 318K+ 임상 이미지 | ★★★ |
| W&B 실험 추적 | (미사용 — 서버 nohup + JSON 로그로 관리) | — |
| DICOM 처리 | (미경험 — PBS JPEG 이미지 사용) | — |

**강점으로 부각할 포인트:**
1. Label noise를 UMAP으로 자동 탐지 → 전문의 재라벨링으로 Precision 32% 향상 (JD 핵심 문제와 정확히 일치)
2. SupCon 표현 학습으로 혼동쌍 분리 44.5× 향상 + 논문 작성 중
3. 실서비스 배포 경험 (FastAPI + Docker + AWS EC2 + ONNX)
4. DDPM 데이터 증강 + 재학습 실험으로 희소 클래스 성능 개선 검증

**약점 / 보완 포인트:**
- W&B 미사용 → "향후 도입 의지" 또는 JSON 기반 실험 추적 경험으로 서술
- DICOM 미경험 → 의료영상 도메인 이해도 + 전환 가능성 강조
- CXR 아닌 PBS → 방법론 전이 가능성 강조 (label noise, semi-supervised, augmentation 모두 CXR에 직접 적용 가능)
