"""
EXP-TS: Two-Stage 재분류 실험
==================================
YOLO(seed_0) 탐지 → SupCon Linear Classifier 재분류 → per-class AP@50 비교

가설: SupCon feature space가 혼동쌍을 더 잘 분리하므로,
     YOLO 탐지 결과를 SupCon으로 재분류하면 AP가 개선될 것이다.

결과: mAP@50  YOLO=0.7791  TS=0.0407  (negative result)
해석: SupCon은 representation 분석 도구로 유효하나,
     실제 detection 재분류에는 적합하지 않음.
     → crop 품질 불균일, 클래스 불균형(RBC 과다 예측) 이 원인.

결과 파일: experiments/results/twostage_reclassification.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from ultralytics import YOLO

# ── 경로 설정 ──────────────────────────────────────────────────────────────
HDD = Path('/home/smile/work/hdd8t')
PBS = Path('/home/smile/work/pbs06')

YOLO_PT   = HDD / 'EXP-006_DeepEnsemble/seed_0/weights/best.pt'
SUPCON_PT = HDD / 'EXP-L4_Analysis/SupCon_HN_s42/best.pt'
FEAT_VAL  = HDD / 'EXP-L4_Analysis/SupCon_HN_s42/features.npy'
LAB_VAL   = HDD / 'EXP-L4_Analysis/SupCon_HN_s42/labels.npy'
TEST_IMG  = PBS / 'test/images'
TEST_LAB  = PBS / 'test/labels'
OUT_JSON  = Path('experiments/results/twostage_reclassification.json')

CLASS_NAMES = [
    'BandNeutrophil', 'Basophil', 'Blast', 'Echinocyte', 'Elliptocyte',
    'Eosinophil', 'GiantPlt', 'Lymphocyte', 'Monocyte', 'Myelocyte',
    'Nucleated', 'PLT', 'RBC', 'Reticulocyte', 'Schistocyte',
    'SegNeutrophil', 'Smudge', 'Stomatocyte', 'TargetCell', 'TearDropCell',
    'ToxicGranule', 'ToxicVacuole', 'clump-PLT', 'degeneWBC', 'hyperSeg.'
]

SUPCON_BATCH = 256
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45
IOU_MATCH    = 0.50


# ── SupCon backbone ────────────────────────────────────────────────────────
class SupConResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)


# ── 유틸 ───────────────────────────────────────────────────────────────────
def iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def load_gt(label_path, img_w, img_h):
    gt = []
    if not Path(label_path).exists():
        return gt
    for line in open(label_path).readlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        gt.append({'cls': cls, 'box': [x1, y1, x2, y2]})
    return gt


def compute_ap(detections, n_gt):
    if n_gt == 0 or not detections:
        return 0.0
    detections.sort(key=lambda x: -x[0])
    tp_cum = fp_cum = 0
    prec, rec = [], []
    for conf, tp in detections:
        if tp:
            tp_cum += 1
        else:
            fp_cum += 1
        prec.append(tp_cum / (tp_cum + fp_cum))
        rec.append(tp_cum / n_gt)
    ap = 0.0
    for k in range(1, len(prec)):
        ap += (rec[k] - rec[k - 1]) * prec[k]
    return ap


def extract_features_batch(crops, model, transform, device):
    tensors = [transform(c) for c in crops]
    all_feats = []
    for i in range(0, len(tensors), SUPCON_BATCH):
        batch_t = torch.stack(tensors[i:i + SUPCON_BATCH]).to(device)
        with torch.no_grad():
            feats = model(batch_t).cpu().numpy()
        all_feats.append(feats)
    return normalize(np.concatenate(all_feats), norm='l2')


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Linear Classifier 학습 (SupCon val features)
    print("\n[1] Linear Classifier 학습...")
    feat_v = normalize(np.load(str(FEAT_VAL)), norm='l2')
    lab_v  = np.load(str(LAB_VAL))
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                             multi_class='multinomial', n_jobs=-1)
    clf.fit(feat_v, lab_v)
    print(f"  val accuracy: {clf.score(feat_v, lab_v):.4f}  (n={len(lab_v)})")

    # 2. SupCon backbone 로드
    print("\n[2] SupCon backbone 로드...")
    ckpt = torch.load(str(SUPCON_PT), map_location=device)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    encoder_state = {}
    for k, v in state.items():
        new_k = None
        if k.startswith('backbone.'):
            new_k = k.replace('backbone.', 'encoder.', 1)
        elif k.startswith('encoder.'):
            new_k = k
        elif k.startswith('module.backbone.'):
            new_k = k.replace('module.backbone.', 'encoder.', 1)
        elif k.startswith('module.encoder.'):
            new_k = k.replace('module.encoder.', 'encoder.', 1)
        if new_k:
            encoder_state[new_k] = v

    supcon_model = SupConResNet().to(device)
    missing, unexpected = supcon_model.load_state_dict(encoder_state, strict=False)
    print(f"  encoder loaded ({len(encoder_state)} keys, missing={len(missing)})")
    supcon_model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. YOLO 탐지 + SupCon 재분류
    print("\n[3] YOLO 탐지 + SupCon 배치 재분류...")
    yolo = YOLO(str(YOLO_PT))
    img_files = sorted(list(TEST_IMG.glob('*.jpg')) + list(TEST_IMG.glob('*.png')))
    print(f"  test images: {len(img_files)}")

    results_yolo = {i: [] for i in range(25)}
    results_ts   = {i: [] for i in range(25)}
    gt_count     = {i: 0  for i in range(25)}

    BATCH = 50
    for i in range(0, len(img_files), BATCH):
        batch = img_files[i:i + BATCH]
        preds = yolo(batch, verbose=False, conf=CONF_THRESH, iou=IOU_THRESH)

        for img_path, pred in zip(batch, preds):
            img = Image.open(str(img_path)).convert('RGB')
            W, H = img.size
            lab_path = TEST_LAB / (img_path.stem + '.txt')
            gt = load_gt(lab_path, W, H)
            for g in gt:
                gt_count[g['cls']] += 1
            gt_matched = [False] * len(gt)

            if pred.boxes is None or len(pred.boxes) == 0:
                continue

            # 모든 detection 정보 수집
            det_info, valid_crops = [], []
            for j in range(len(pred.boxes)):
                x1, y1, x2, y2 = [int(v) for v in pred.boxes.xyxy[j].tolist()]
                conf     = float(pred.boxes.conf[j])
                yolo_cls = int(pred.boxes.cls[j])
                best_iou, best_gi = 0, -1
                for gi, g in enumerate(gt):
                    v = iou([x1, y1, x2, y2], g['box'])
                    if v > best_iou:
                        best_iou, best_gi = v, gi
                tp_yolo = (best_iou >= IOU_MATCH and not gt_matched[best_gi]
                           and gt[best_gi]['cls'] == yolo_cls) if best_gi >= 0 else False
                results_yolo[yolo_cls].append((conf, int(tp_yolo)))
                crop  = img.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
                valid = crop.width >= 8 and crop.height >= 8
                det_info.append({'conf': conf, 'yolo_cls': yolo_cls,
                                 'best_iou': best_iou, 'best_gi': best_gi,
                                 'tp_yolo': tp_yolo, 'valid': valid})
                if valid:
                    valid_crops.append(crop)

            # 배치 SupCon 추론
            feats    = extract_features_batch(valid_crops, supcon_model, transform, device) \
                       if valid_crops else np.zeros((0, 2048))
            ts_preds = clf.predict(feats).astype(int) if len(feats) > 0 else []

            crop_idx = 0
            for d in det_info:
                if not d['valid']:
                    results_ts[d['yolo_cls']].append((d['conf'], int(d['tp_yolo'])))
                    continue
                ts_cls  = int(ts_preds[crop_idx]); crop_idx += 1
                best_gi = d['best_gi']
                tp_ts   = (d['best_iou'] >= IOU_MATCH and not gt_matched[best_gi]
                           and gt[best_gi]['cls'] == ts_cls) if best_gi >= 0 else False
                results_ts[ts_cls].append((d['conf'], int(tp_ts)))
                if (d['tp_yolo'] or tp_ts) and best_gi >= 0:
                    gt_matched[best_gi] = True

        if (i // BATCH) % 5 == 0:
            print(f"  {i + len(batch)}/{len(img_files)} done")

    # 4. AP@50 계산
    print("\n[4] AP@50 계산...")
    summary = {}
    for ci, name in enumerate(CLASS_NAMES):
        ap_y = compute_ap(results_yolo[ci], gt_count[ci])
        ap_t = compute_ap(results_ts[ci],   gt_count[ci])
        summary[name] = {'yolo_ap50': round(ap_y, 4), 'ts_ap50': round(ap_t, 4),
                         'delta': round(ap_t - ap_y, 4), 'gt_n': gt_count[ci]}
        print(f"  {name}: YOLO={ap_y:.4f} -> TS={ap_t:.4f} ({ap_t - ap_y:+.4f})")

    overall_y = sum(v['yolo_ap50'] for v in summary.values()) / 25
    overall_t = sum(v['ts_ap50']   for v in summary.values()) / 25
    summary['_overall'] = {'yolo_ap50': round(overall_y, 4),
                           'ts_ap50':   round(overall_t, 4),
                           'delta':     round(overall_t - overall_y, 4)}
    print(f"\nmAP50  YOLO={overall_y:.4f}  TS={overall_t:.4f}  delta={overall_t - overall_y:+.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n결과 저장: {OUT_JSON}")


if __name__ == '__main__':
    main()
