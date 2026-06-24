"""
PBS Inference Server — API Client Demo
----------------------------------------
FastAPI 추론 서버(/inf 엔드포인트)에 이미지를 전송하고 결과를 시각화합니다.
production_reference/inference_server/main.py 가 실행 중일 때 사용합니다.

Usage:
    python examples/api_client.py --image path/to/image.jpg
    python examples/api_client.py --image path/to/image.jpg --server http://localhost:8000
    python examples/api_client.py --image path/to/image.jpg --save result.jpg
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

DEFAULT_SERVER = "http://localhost:8000"
DEFAULT_MODEL  = "yolov11s_pbs"

CLASS_COLORS = {
    # WBC
    "Neutrophil": (231, 76,  60),
    "BandNeutrophil": (192, 57, 43),
    "SegNeutrophil":  (231, 76, 60),
    "Lymphocyte":     (52, 152, 219),
    "Monocyte":       (41, 128, 185),
    "Eosinophil":     (155, 89, 182),
    "Basophil":       (142, 68, 173),
    "Blast":          (230, 126, 34),
    "Myelocyte":      (211, 84,  0),
    # RBC
    "RBC":            (46, 204, 113),
    "Schistocyte":    (39, 174,  96),
    "Echinocyte":     (26, 188, 156),
    "TargetCell":     (22, 160, 133),
    # PLT
    "PLT":            (243, 156, 18),
    "GiantPlt":       (241, 196, 15),
}

DEFAULT_COLOR = (149, 165, 166)


def image_to_base64(image_path: str) -> tuple[str, int, int]:
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] 이미지를 읽을 수 없습니다: {image_path}")
        sys.exit(1)
    h, w = img.shape[:2]
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", w, h


def call_inference_api(server: str, model_name: str, image_path: str) -> dict:
    b64_image, width, height = image_to_base64(image_path)
    payload = {
        "model_name": model_name,
        "width": width,
        "height": height,
        "image": b64_image,
    }
    print(f"[>] POST {server}/inf  (image: {width}x{height})")
    try:
        resp = requests.post(f"{server}/inf", json=payload, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"[!] 서버에 연결할 수 없습니다: {server}")
        print("    production_reference/inference_server/main.py 를 먼저 실행하세요.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[!] HTTP 오류: {e}")
        sys.exit(1)

    data = resp.json()
    if isinstance(data, str):
        data = json.loads(data)
    return data


def visualize(image_path: str, detections: list, save_path: str = None):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(f"PBS Cell Detection — {len(detections)} detections", fontsize=13, fontweight="bold")

    counts = {}
    for det in detections:
        cls   = det["class"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        prob  = det["prob"]
        color_bgr = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        color_hex = "#{:02X}{:02X}{:02X}".format(*color_bgr)

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color_hex, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 4, 0), f"{cls} {prob:.2f}",
            color="white", fontsize=6.5, fontweight="bold",
            bbox=dict(facecolor=color_hex, alpha=0.8, pad=1, edgecolor="none")
        )
        counts[cls] = counts.get(cls, 0) + 1

    summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9, color="#555")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] 결과 저장: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PBS API Client Demo")
    parser.add_argument("--image",  required=True, help="입력 이미지 경로")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"서버 주소 (기본값: {DEFAULT_SERVER})")
    parser.add_argument("--model",  default=DEFAULT_MODEL,  help=f"모델 이름 (기본값: {DEFAULT_MODEL})")
    parser.add_argument("--save",   default=None,           help="결과 이미지 저장 경로")
    parser.add_argument("--json",   action="store_true",    help="JSON 결과만 출력")
    args = parser.parse_args()

    result = call_inference_api(args.server, args.model, args.image)

    if result.get("result_code") != "OK":
        print(f"[!] 추론 오류:\n{result.get('error_trace', result)}")
        sys.exit(1)

    detections = result["result"]
    print(f"[✓] 탐지 완료 — {len(detections)}개")

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    counts = {}
    for det in detections:
        counts[det["class"]] = counts.get(det["class"], 0) + 1
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:20s}: {cnt}")

    visualize(args.image, detections, save_path=args.save)


if __name__ == "__main__":
    main()
