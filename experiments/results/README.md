# Experiment Results

실험 결과 JSON 파일 모음. 각 파일은 재현 없이 결과를 확인할 수 있도록 저장.

| 파일 | 실험 | 핵심 수치 |
|------|------|-----------|
| `ddpm_augmentation.json` | DDPM 희소 클래스 증강 후 YOLOv11s 재학습 (3조건) | Baseline 0.8794 → Aug_1000 **0.8821** |
| `twostage_reclassification.json` | YOLO + SupCon 두 스테이지 재분류 | mAP50 0.7791 → **0.0407** (negative result) |

## ddpm_augmentation.json

3조건 비교: `Baseline` / `Aug_500` (DDPM +500/class) / `Aug_1000` (DDPM +1000/class)

```json
{
  "Baseline":  { "mAP50": 0.8794, "per_class": {...} },
  "Aug_500":   { "mAP50": 0.8811, "per_class": {...} },
  "Aug_1000":  { "mAP50": 0.8821, "per_class": {...} }
}
```

희소 클래스 개선 (Aug_1000 기준):
- Atypical: 0.759 → **0.787** (+2.8%)
- Nucleated: 0.868 → **0.884** (+1.6%)
- TargetCell: 0.659 → **0.665** (+0.6%)

## twostage_reclassification.json

YOLO 탐지 결과를 SupCon Linear Classifier로 재분류했을 때 AP@50 변화.

```json
{
  "RBC":  { "yolo_ap50": 0.9514, "ts_ap50": 0.5386, "delta": -0.4129 },
  "_overall": { "yolo_ap50": 0.7791, "ts_ap50": 0.0407, "delta": -0.7384 }
}
```

**결론:** SupCon은 표현 분석(UMAP, linear probe)에 적합하나 탐지 대체는 부적합.  
crop 품질 불균일 + 클래스 불균형(RBC 63K)이 원인.
