#!/usr/bin/env python3
import sys, json, cv2
from pathlib import Path
from ultralytics import YOLO
import ultralytics.utils.patches as _patches
import ultralytics.data.base as _dbase

# cv2 monkey-patch (numpy 2.x / cv2 4.7 호환)
def _cv2_imread(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imread(str(filename), flags)
_patches.imread = _cv2_imread
_dbase.imread = _cv2_imread

AUG_ROOT   = Path("/home/smile/work/hdd8t/EXP-G3_Augmented")
OUT_ROOT   = Path("/home/smile/work/hdd8t/EXP-G4_YOLOv11")
OUT_ROOT.mkdir(exist_ok=True)
LOG_FILE   = OUT_ROOT / "training_log.json"

# Use local yolo11s.pt if available
WEIGHTS_CANDIDATES = [
    Path("/home/smile/work/a_CKM_mo/ObjectDetection/YOLO/yolo11s.pt"),
    Path("/home/smile/work/hdd8t/yolo11s.pt"),
]
PRETRAINED = "yolo11s.pt"
for p in WEIGHTS_CANDIDATES:
    if p.exists():
        PRETRAINED = str(p)
        break

print(f"Using weights: {PRETRAINED}", flush=True)

conditions = ["Baseline", "Aug_500", "Aug_1000"]
results = {}

for cond in conditions:
    yaml_path = AUG_ROOT / cond / "data.yaml"
    if not yaml_path.exists():
        print(f"[{cond}] YAML not found: {yaml_path}", flush=True)
        results[cond] = {"error": "yaml not found"}
        continue

    print(f"\n[{cond}] Training start...", flush=True)
    try:
        model = YOLO(PRETRAINED)
        model.train(
            data=str(yaml_path),
            project=str(OUT_ROOT),
            name=cond,
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            device="cuda:1",
            amp=False,
            workers=0,
            exist_ok=True,
            verbose=True,
        )
        metrics = model.val(data=str(yaml_path), split="test", device="cuda:1")
        mAP50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0))
        print(f"[{cond}] mAP50={mAP50:.4f}", flush=True)

        ap_class = {}
        cls_names = ["Neutrophil","Lymphocyte","Monocyte","Eosinophil","Basophil",
                     "BandNeutrophil","SegNeutrophil","hyperSeg","NormalRBC","Microcyte",
                     "Macrocyte","Hypochromia","Schistocyte","Spherocyte","TargetCell",
                     "Stomatocyte","Acanthocyte","Echinocyte","Nucleated","Blast",
                     "Prolymphocyte","Atypical","Platelet","GiantPlt","Thrombocytopenia"]
        if hasattr(metrics, "ap_class_index") and hasattr(metrics.box, "ap50"):
            for i, ci in enumerate(metrics.ap_class_index):
                if 0 <= ci < len(cls_names):
                    ap_class[cls_names[ci]] = round(float(metrics.box.ap50[i]), 4)

        results[cond] = {"mAP50": round(mAP50, 4), "per_class": ap_class, "status": "done"}

    except Exception as e:
        print(f"[{cond}] ERROR: {e}", flush=True)
        results[cond] = {"error": str(e)}

    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{cond}] Results saved: {LOG_FILE}", flush=True)

print("\n=== All training complete ===", flush=True)
print(json.dumps(results, indent=2), flush=True)
