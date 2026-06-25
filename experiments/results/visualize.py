"""
실험 결과 시각화 — 서버·GPU 불필요
=====================================
experiments/results/ 폴더의 JSON 파일만 읽어 결과를 그래프로 출력합니다.

Usage:
    python experiments/results/visualize.py            # 모든 그래프 표시
    python experiments/results/visualize.py --save     # PNG 파일로 저장
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).parent
DDPM_JSON      = HERE / "ddpm_augmentation.json"
TWOSTAGE_JSON  = HERE / "twostage_reclassification.json"
SAVE_DIR       = HERE / "figures"

COLORS = {"Baseline": "#95A5A6", "Aug_500": "#3498DB", "Aug_1000": "#E74C3C"}


# ── 1. DDPM 증강 결과 ──────────────────────────────────────────────────────
def plot_ddpm(save=False):
    with open(DDPM_JSON) as f:
        data = json.load(f)

    conditions = ["Baseline", "Aug_500", "Aug_1000"]
    classes    = [c for c in data["Baseline"]["per_class"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("DDPM Augmentation — YOLOv11s Re-training Results", fontsize=14, fontweight="bold")

    # ── 왼쪽: 전체 mAP 비교
    ax = axes[0]
    maps = [data[c]["mAP50"] for c in conditions]
    bars = ax.bar(conditions, maps, color=[COLORS[c] for c in conditions], width=0.5, edgecolor="white")
    ax.set_ylim(0.87, 0.895)
    ax.set_ylabel("mAP@50")
    ax.set_title("Overall mAP@50 by Condition")
    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(maps[0], color=COLORS["Baseline"], linestyle="--", alpha=0.4)

    # ── 오른쪽: 클래스별 AP 변화 (Aug_1000 - Baseline, 상위 10 변화)
    ax2 = axes[1]
    deltas = {c: data["Aug_1000"]["per_class"][c] - data["Baseline"]["per_class"][c]
              for c in classes}
    sorted_cls = sorted(deltas, key=lambda x: abs(deltas[x]), reverse=True)[:12]
    vals  = [deltas[c] for c in sorted_cls]
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in vals]
    ax2.barh(sorted_cls[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("AP@50 Change (Aug_1000 − Baseline)")
    ax2.set_title("Per-class AP Change (Top-12 by magnitude)")
    ax2.set_xlim(-0.04, 0.04)

    plt.tight_layout()
    if save:
        SAVE_DIR.mkdir(exist_ok=True)
        plt.savefig(SAVE_DIR / "ddpm_results.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {SAVE_DIR}/ddpm_results.png")
    else:
        plt.show()
    plt.close()


# ── 2. Two-Stage 재분류 결과 ───────────────────────────────────────────────
def plot_twostage(save=False):
    with open(TWOSTAGE_JSON) as f:
        data = json.load(f)

    classes = [k for k in data if k != "_overall"]
    yolo_ap = [data[c]["yolo_ap50"] for c in classes]
    ts_ap   = [data[c]["ts_ap50"]   for c in classes]
    gt_n    = [data[c]["gt_n"]      for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Two-Stage Reclassification: YOLO → SupCon Linear Classifier\n"
        f"Overall  YOLO mAP@50={data['_overall']['yolo_ap50']:.4f}  "
        f"TS mAP@50={data['_overall']['ts_ap50']:.4f}  "
        f"Δ={data['_overall']['delta']:+.4f}",
        fontsize=13, fontweight="bold"
    )

    # ── 왼쪽: YOLO vs TS per-class AP 산점도
    ax = axes[0]
    ax.scatter(yolo_ap, ts_ap, s=[max(20, n / 500) for n in gt_n],
               alpha=0.7, c="#E74C3C", edgecolors="white", linewidths=0.5)
    for i, cls in enumerate(classes):
        if abs(ts_ap[i] - yolo_ap[i]) > 0.3 or ts_ap[i] > 0.3:
            ax.annotate(cls, (yolo_ap[i], ts_ap[i]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    lim = [0, 1.05]
    ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("YOLO AP@50")
    ax.set_ylabel("Two-Stage AP@50")
    ax.set_title("Per-class AP: YOLO vs Two-Stage\n(bubble size ∝ GT count)")
    ax.set_xlim(*lim); ax.set_ylim(*lim)

    # ── 오른쪽: 클래스별 AP 나란히 막대
    ax2 = axes[1]
    x = np.arange(len(classes))
    w = 0.4
    ax2.bar(x - w/2, yolo_ap, w, label="YOLO",      color="#3498DB", alpha=0.85)
    ax2.bar(x + w/2, ts_ap,   w, label="Two-Stage", color="#E74C3C", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=55, ha="right", fontsize=7)
    ax2.set_ylabel("AP@50")
    ax2.set_title("Per-class AP@50 Comparison")
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    if save:
        SAVE_DIR.mkdir(exist_ok=True)
        plt.savefig(SAVE_DIR / "twostage_results.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {SAVE_DIR}/twostage_results.png")
    else:
        plt.show()
    plt.close()


# ── 3. 연구 흐름 요약표 ────────────────────────────────────────────────────
def plot_research_summary(save=False):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    fig.suptitle("PBS Blood Cell AI — Research Summary", fontsize=14, fontweight="bold")

    rows = [
        ["실험",               "방법",                                   "핵심 결과"],
        ["YOLO 모델 비교",     "YOLOv5 → v8 → v10 → v11s",             "mAP@50 0.879 달성"],
        ["Label Noise 처리",   "UMAP 경계 탐지 + 전문의 재라벨링",      "Precision 0.68 → 0.90 (+32%)"],
        ["Deep Ensemble",      "5 seeds × YOLOv11s + WBF",              "재현성 확보, FP 감소"],
        ["캘리브레이션",       "Temperature Scaling + Conformal Pred",   "ECE 감소, 커버리지 보장"],
        ["EigenCAM (v1→v7)",   "FPN layer별 heat map, 11개 혼동쌍",     "오분류 원인 규명"],
        ["SupCon 표현 학습",   "ResNet50 + Contrastive Loss",           "Silhouette 0.112→0.500, 혼동쌍 44.5×"],
        ["DDPM 증강",          "희소 5 class 합성 (FID≈76)",            "전체 +0.27%, Atypical +2.8%"],
        ["VAE OOD 필터",       "ResidualBlock+VAE+UNet, 3σ threshold",  "FP 감소 (블러·이물질 제거)"],
        ["Two-Stage (negative)","YOLO + SupCon 재분류",                 "mAP 0.779→0.041 (실패, 원인 분석 완료)"],
    ]

    col_widths = [0.18, 0.42, 0.40]
    row_colors_odd  = "#F8F9FA"
    row_colors_even = "#FFFFFF"
    header_color    = "#2C3E50"

    y = 0.95
    dy = 0.083
    for ri, row in enumerate(rows):
        x = 0.01
        bg = header_color if ri == 0 else (row_colors_odd if ri % 2 == 1 else row_colors_even)
        fc = "white" if ri == 0 else "black"
        fw = "bold" if ri == 0 else "normal"
        rect = mpatches.FancyBboxPatch((x - 0.005, y - dy * 0.85), 0.99, dy * 0.88,
                                       boxstyle="square,pad=0", facecolor=bg, edgecolor="#DEE2E6",
                                       linewidth=0.5, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            ax.text(x + 0.005, y - dy * 0.35, cell,
                    transform=ax.transAxes, fontsize=8.5,
                    color=fc, fontweight=fw, va="center")
            x += cw
        y -= dy

    plt.tight_layout()
    if save:
        SAVE_DIR.mkdir(exist_ok=True)
        plt.savefig(SAVE_DIR / "research_summary.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {SAVE_DIR}/research_summary.png")
    else:
        plt.show()
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험 결과 시각화")
    parser.add_argument("--save", action="store_true", help="PNG 파일로 저장")
    args = parser.parse_args()

    print("[1/3] DDPM 증강 결과...")
    plot_ddpm(save=args.save)

    print("[2/3] Two-Stage 재분류 결과...")
    plot_twostage(save=args.save)

    print("[3/3] 연구 요약표...")
    plot_research_summary(save=args.save)

    print("완료.")
