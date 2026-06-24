# Inference Server — 실서비스 배포 코드

U2Bio PBS AI 추론 서버. Docker + FastAPI + AWS EC2 기반.

## 파일 설명

| 파일 | 설명 |
|------|------|
| `main.py` | FastAPI 엔트리포인트. `/predict` POST 엔드포인트, 이미지 입력 → 탐지 결과 JSON 반환 |
| `u2_utils.py` | YOLOv11s 추론 파이프라인. 이미지 전처리, 타일링, WBF, SupCon 재분류, DB 로깅 포함 |
| `copymodel.sh` | S3/로컬에서 모델 파일(.onnx, .pt) 컨테이너로 복사하는 배포 스크립트 |
| `Dockerfile` | 서비스 컨테이너. `FROM u2python:latest`, port 8000, uvicorn |
| `Dockerfile_base` | 베이스 이미지. Python 3.10 + PyTorch + CUDA + FastAPI |

## 배포 스택

```
Client → FastAPI (uvicorn) → YOLOv11s ONNX → SupCon Reclassifier → MariaDB
         Docker container    ThreadPoolExecutor (8 workers)
         AWS EC2 (A100)
```

## 주요 기능 (`u2_utils.py`)

- **이미지 타일링**: 고해상도 PBS 슬라이드 → 겹치는 타일로 분할
- **YOLOv11s 탐지**: 25-class bounding box 탐지
- **WBF**: 타일 경계 중복 탐지 Weighted Boxes Fusion
- **SupCon 재분류**: 혼동쌍(Band/Seg Neutrophil 등) 재분류
- **VAE OOD 필터링**: reconstruction error로 blur/artifact 이미지 제거
- **DB 로깅**: 탐지 결과 MariaDB 저장 (JWT 인증)
