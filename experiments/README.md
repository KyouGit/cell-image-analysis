# Experiments — 서버 실험 코드 아카이브

U2Bio PBS 프로젝트 서버에서 수행한 실험 스크립트들. 연구 진행 과정을 보여주는 포트폴리오용 코드.

## 폴더 구조

```
experiments/
├── eigencam/               # XAI — EigenCAM 분석 (버전별 진화)
│   ├── run_eigencam_v1.py  #   초기: FPN P3/P4/P5 heat map 생성
│   ├── run_eigencam_v4.py  #   혼동쌍 8종 한 번에 분석
│   ├── run_eigencam_v6.py  #   전체 25-class 배치 처리
│   ├── run_eigencam_v7.py  #   최종: WBC×4, RBC×4, PLT×2 — 논문 제출본
│   ├── run_schisto_cam_candidates.py  # Schistocyte 후보 CAM 분석
│   └── run_schistocyte_only.py        # Schistocyte 단독 탐지 실험
│
├── training/               # 모델 학습 스크립트
│   ├── ensemble_train.py   #   Deep Ensemble (5 seeds) 자동화
│   ├── run_yolo_train.py   #   YOLOv11s 기본 학습
│   └── run_yolo_train2.py  #   YOLOv11s — Aug/Hyperparams 튜닝
│
└── archive/                # 버전 이력 (진화 과정)
    ├── pipeline_v1.py      #   초기 PBS 파이프라인 (YOLOv5 기반)
    ├── api_v1/app.py       #   최초 배포 API (YOLOv5 + Flask)
    ├── api_v2/             #   v2 API (YOLOv8 + FastAPI)
    │   ├── app.py
    │   ├── u2_utils.py
    │   └── u2_imagesplitter.py
    └── api_dev/u2_utils.py #   dev 브랜치 — SupCon 재분류 실험 중
```

## 연구 진화 요약

| 단계 | 모델 | 특이사항 |
|------|------|---------|
| v1 (archive/api_v1) | YOLOv5 | 3-class 탐지, Flask 서빙 |
| v2 (archive/api_v2) | YOLOv8 | 25-class, FastAPI, u2_imagesplitter |
| v3 (production_reference) | YOLOv11s | Soft-label + Deep Ensemble + WBF |
| dev (archive/api_dev) | YOLOv11s + SupCon | SupCon 재분류기 실험 중 |

## 핵심 실험 파일

- **`eigencam/run_eigencam_v7.py`**: 논문 Figure 8–11 생성 코드. FPN P2~P5 각 레이어 히트맵 + D-RISE saliency divergence 정량화.
- **`training/ensemble_train.py`**: seed 0–4 자동 순차 학습, 결과 JSON 저장.
- **`training/run_yolo_train2.py`**: Mosaic augmentation × 1000 epoch, EXP-G4 계열.
