"""
PBS Blood Cell Detection — Local Demo
--------------------------------------
BCCD 데이터셋 이미지를 입력받아 YOLOv8 탐지 결과를 시각화합니다.
서버 없이 로컬에서 전체 파이프라인을 확인하는 용도입니다.

Usage:
    python examples/demo.py                        # 랜덤 샘플 이미지 사용
    python examples/demo.py --image path/to/img.jpg
    python examples/demo.py --save result.jpg      # 결과 이미지 저장
"""

import argparse
import sys
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

WEIGHTS = "pipeline/models/yolov8n_bccd.pt"   # pipeline/yolo_detect.py 로 학습된 모델

CLASS_COLORS = {
    "WBC":      "#E74C3C",
    "RBC":      "#3498DB",
    "Platelet": "#2ECC71",
}

SAMPLE_DIR = Path("datasets/BCCD/images/test")


def load_model():
    pt = Path(WEIGHTS)
    if not pt.exists():
        print(f"[!] 가중치 없음: {pt}")
        print("    먼저 'python pipeline/yolo_detect.py --mode train' 을 실행하세요.")
        sys.exit(1)
    return YOLO(str(pt))


def pick_sample_image():
    if not SAMPLE_DIR.exists():
        print(f"[!] 샘플 이미지 폴더 없음: {SAMPLE_DIR}")
        print("    먼저 'python pipeline/prepare_data.py' 를 실행하세요.")
        sys.exit(1)
    images = list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.png"))
    if not images:
        print(f"[!] {SAMPLE_DIR} 에 이미지가 없습니다.")
        sys.exit(1)
    return random.choice(images)


def run_inference(model, image_path):
    results = model.predict(source=str(image_path), conf=0.25, iou=0.45, verbose=False)
    return results[0]


def draw_results(image_path, result, save_path=None):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("PBS Blood Cell Detection — YOLOv8n (BCCD Demo)", fontsize=14, fontweight="bold")

    # 왼쪽: 원본 이미지
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image", fontsize=11)
    axes[0].axis("off")

    # 오른쪽: 탐지 결과
    axes[1].imshow(img_rgb)
    axes[1].set_title("Detection Result", fontsize=11)
    axes[1].axis("off")

    counts = {}
    names = result.names
    boxes = result.boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        color = CLASS_COLORS.get(cls_name, "#F39C12")

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, max(y1 - 4, 0), f"{cls_name} {conf:.2f}",
            color="white", fontsize=6, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor="none")
        )
        counts[cls_name] = counts.get(cls_name, 0) + 1

    summary = "  |  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    fig.text(0.5, 0.02, f"Total detections: {len(boxes)}   ({summary})", ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] 결과 저장: {save_path}")
    else:
        plt.show()

    plt.close()
    return counts


def print_json_result(result):
    names = result.names
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        conf = round(float(box.conf[0]), 4)
        cls_name = names[int(box.cls[0])]
        detections.append({"class": cls_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "prob": conf})

    import json
    print("\n── API Response (same format as production /inf endpoint) ──")
    print(json.dumps({"result_code": "OK", "result": detections[:5]}, indent=2))
    if len(detections) > 5:
        print(f"  ... and {len(detections) - 5} more detections")


def main():
    parser = argparse.ArgumentParser(description="PBS Cell Detection Demo")
    parser.add_argument("--image", type=str, default=None, help="입력 이미지 경로")
    parser.add_argument("--save",  type=str, default=None, help="결과 이미지 저장 경로")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else pick_sample_image()
    print(f"[>] 입력 이미지: {image_path}")

    model = load_model()
    print("[>] 추론 중...")
    result = run_inference(model, image_path)

    counts = draw_results(image_path, result, save_path=args.save)
    print_json_result(result)

    print(f"\n[✓] 탐지 완료 — {sum(counts.values())}개 검출 ({counts})")


if __name__ == "__main__":
    main()
