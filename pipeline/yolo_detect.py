"""
Blood Cell Detection using YOLOv8

BCCD 데이터셋으로 YOLOv8 Object Detection 학습 → 세포 크롭 추출

Usage:
    python yolo_detect.py --mode train    # 학습
    python yolo_detect.py --mode eval     # 평가
    python yolo_detect.py --mode crop     # 세포 크롭 추출
    python yolo_detect.py --mode all      # 전체 파이프라인
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


# Configuration
DATA_YAML = 'data/bccd_yolo/data.yaml'
CLASS_NAMES = ['WBC', 'RBC', 'Platelet']
CROP_OUTPUT_DIR = 'data/cropped_cells'


def train(data_yaml=DATA_YAML, epochs=100, imgsz=640, batch=16, patience=20):
    """Train YOLOv8 detection model on BCCD dataset"""
    print("=" * 70)
    print("YOLOv8 Detection Training")
    print("=" * 70)

    model = YOLO('yolov8n.pt')
    print(f"[INFO] Model: YOLOv8n (detection)")
    print(f"[INFO] Data: {data_yaml}")
    print(f"[INFO] Epochs: {epochs}, ImgSz: {imgsz}, Batch: {batch}, Patience: {patience}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        workers=0,
        project='runs',
        name='detect_bccd',
        exist_ok=True,
        verbose=True,
    )

    print(f"\n[INFO] Training complete. Best model: runs/detect/runs/detect/bccd/weights/best.pt")
    return model


def evaluate(model_path='runs/detect/runs/detect/bccd/weights/best.pt', data_yaml=DATA_YAML):
    """Evaluate trained model and print per-class metrics"""
    print("=" * 70)
    print("YOLOv8 Detection Evaluation")
    print("=" * 70)

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split='test')

    print(f"\n[RESULTS]")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")

    # Per-class AP
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        print(f"\n  Per-class AP@50:")
        for i, ap in enumerate(metrics.box.ap50):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'
            print(f"    {name}: {ap:.4f}")

    return metrics


def crop_cells(
    model_path='runs/detect/runs/detect/bccd/weights/best.pt',
    source_dir='data/bccd_yolo/images',
    output_dir=CROP_OUTPUT_DIR,
    conf_threshold=0.25,
    padding_ratio=0.1,
    min_size=20,
):
    """
    Crop detected cells from images using trained YOLO model

    Args:
        model_path: Path to trained YOLO weights
        source_dir: Directory containing images (scans all subdirs)
        output_dir: Output directory for cropped cells
        conf_threshold: Minimum confidence threshold
        padding_ratio: Padding around bbox as ratio of bbox size
        min_size: Minimum crop size in pixels (skip tiny detections)
    """
    print("=" * 70)
    print("Cell Cropping from YOLO Detections")
    print("=" * 70)

    model = YOLO(model_path)
    output_dir = Path(output_dir)

    # Create output directories
    for cls_name in CLASS_NAMES:
        (output_dir / cls_name).mkdir(parents=True, exist_ok=True)

    # Collect all images from source directory
    source_dir = Path(source_dir)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(source_dir.rglob(ext))
    image_paths = sorted(set(image_paths))  # deduplicate

    print(f"[INFO] Found {len(image_paths)} images")
    print(f"[INFO] Conf threshold: {conf_threshold}, Padding: {padding_ratio}, Min size: {min_size}")

    crop_counts = {name: 0 for name in CLASS_NAMES}
    total_skipped = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        # Run detection
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        for j in range(len(boxes)):
            cls_id = int(boxes.cls[j].item())
            conf = float(boxes.conf[j].item())

            if cls_id >= len(CLASS_NAMES):
                continue

            cls_name = CLASS_NAMES[cls_id]

            # Get bbox coordinates (xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)

            # Compute padding
            bw = x2 - x1
            bh = y2 - y1

            if bw < min_size or bh < min_size:
                total_skipped += 1
                continue

            pad_x = int(bw * padding_ratio)
            pad_y = int(bh * padding_ratio)

            # Apply padding with bounds checking
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w_img, x2 + pad_x)
            y2_pad = min(h_img, y2 + pad_y)

            # Crop
            crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                total_skipped += 1
                continue

            # Save crop
            crop_name = f"{img_path.stem}_{j:03d}_conf{conf:.2f}.jpg"
            crop_path = output_dir / cls_name / crop_name
            cv2.imwrite(str(crop_path), crop)
            crop_counts[cls_name] += 1

    # Print summary
    print(f"\n[RESULTS] Cropped cells saved to {output_dir}")
    total = 0
    for cls_name, count in crop_counts.items():
        print(f"  {cls_name}: {count} crops")
        total += count
    print(f"  Total: {total} crops")
    if total_skipped > 0:
        print(f"  Skipped (too small): {total_skipped}")

    return crop_counts


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Blood Cell Detection')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'eval', 'crop', 'all'],
                        help='Execution mode')
    parser.add_argument('--data', type=str, default=DATA_YAML,
                        help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='runs/detect/runs/detect/bccd/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for cropping')
    parser.add_argument('--padding', type=float, default=0.1,
                        help='Padding ratio around bbox')
    args = parser.parse_args()

    if args.mode in ('train', 'all'):
        train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
        )

    if args.mode in ('eval', 'all'):
        model_path = args.model
        if args.mode == 'all':
            model_path = 'runs/detect/runs/detect/bccd/weights/best.pt'
        evaluate(model_path=model_path, data_yaml=args.data)

    if args.mode in ('crop', 'all'):
        model_path = args.model
        if args.mode == 'all':
            model_path = 'runs/detect/runs/detect/bccd/weights/best.pt'
        crop_cells(
            model_path=model_path,
            source_dir='data/bccd_yolo/images',
            output_dir=CROP_OUTPUT_DIR,
            conf_threshold=args.conf,
            padding_ratio=args.padding,
        )


if __name__ == '__main__':
    main()
