"""EXP-006: Deep Ensemble - 5x YOLOv11s sequential seeds"""
import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLO

DATA_YAML = "/home/smile/work/pbs06/yolov5/data/blood_cell.yaml"
YOLO_PT   = "/home/smile/work/a_CKM_mo/ObjectDetection/YOLO/yolo11s.pt"
OUT_BASE  = "/home/smile/work/hdd8t/EXP-006_DeepEnsemble"
LOG_PATH  = f"{OUT_BASE}/train.log"
SEEDS     = [0, 42, 100, 1234, 9999]

os.makedirs(OUT_BASE, exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

results = {}
log(f"=== Deep Ensemble START | seeds={SEEDS} ===")
log(f"data: {DATA_YAML}")
log(f"output: {OUT_BASE}")

for seed in SEEDS:
    log(f"\n--- seed={seed} START ---")
    try:
        model = YOLO(YOLO_PT)
        r = model.train(
            data=DATA_YAML,
            epochs=800,
            patience=100,
            imgsz=384,
            batch=64,
            optimizer="AdamW",
            lr0=0.00364,
            lrf=0.000643,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            project=OUT_BASE,
            name=f"seed_{seed}",
            seed=seed,
            exist_ok=False,
            verbose=False,
        )
        best_map = float(r.results_dict.get("metrics/mAP50(B)", 0))
        best_path = str(r.save_dir / "weights/best.pt")
        results[f"seed_{seed}"] = {
            "status": "done",
            "mAP50": best_map,
            "weights": best_path,
        }
        log(f"seed={seed} DONE | mAP50={best_map:.4f} | weights={best_path}")
    except Exception as e:
        import traceback
        log(f"seed={seed} ERROR: {e}")
        log(traceback.format_exc())
        results[f"seed_{seed}"] = {"status": "error", "error": str(e)}

with open(f"{OUT_BASE}/results.json", "w") as f:
    json.dump(results, f, indent=2)
log("=== All seeds done ===")
