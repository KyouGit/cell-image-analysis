"""
EXP-L3b: Supervised Contrastive Learning — Proper Hard Negative Weighting
==========================================================================
EXP-L3 (기존): SupConLoss.__init__에서 hard_pairs 정의했지만 forward()에서 미사용
               → 사실상 standard SupCon + CLASS_NAMES 오류 (dataset 순서와 불일치)
EXP-L3b (본 실험):
  - 데이터셋 라벨 순서와 일치하는 올바른 CLASS_NAMES (YOLO 학습 클래스 순서)
  - CORRECTED_HARD_PAIRS: 논문 Appendix A.2 기준 10쌍 (WBC 5쌍 + RBC 5쌍)
  - SupConLoss.forward()에서 hard_pairs로 분모 가중치 실제 적용 (hard_weight=2.0)
출력: /home/smile/work/hdd8t/EXP-L3b_SupCon_HN/
"""
import requests, json, time, websocket, uuid, re

SERVER = "http://211.232.120.231:8889"
s = requests.Session()
s.get(SERVER)
xsrf = s.cookies.get("_xsrf", "")
headers = {"X-XSRFToken": xsrf, "Content-Type": "application/json"}

r = s.post(f"{SERVER}/api/kernels", headers=headers, json={"name": "python3"})
kid = r.json()["id"]
print(f"kernel: {kid}")
time.sleep(2)

code = r'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)

OUT_DIR = Path('/home/smile/work/hdd8t/EXP-L3b_SupCon_HN')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 데이터셋 라벨 순서와 일치하는 CLASS_NAMES (YOLO 학습 클래스 ID 0-24)
CLASS_NAMES = [
    'BandNeutrophil', 'Basophil', 'Blast', 'Echinocyte', 'Elliptocyte',
    'Eosinophil', 'GiantPlt', 'Lymphocyte', 'Monocyte', 'Myelocyte',
    'Nucleated', 'PLT', 'RBC', 'Reticulocyte', 'Schistocyte',
    'SegNeutrophil', 'Smudge', 'Stomatocyte', 'TargetCell', 'TearDropCell',
    'ToxicGranule', 'ToxicVacuole', 'clump-PLT', 'degeneWBC', 'hyperSeg.'
]
N_CLS = len(CLASS_NAMES)
print(f'N_CLS={N_CLS}', flush=True)

# Hard Negative Pairs (논문 Appendix A.2 기준)
# WBC 5쌍: 성숙 연속체 + 형태학적 유사 쌍
# RBC 5쌍: 이형 적혈구 스펙트럼
CORRECTED_HARD_PAIRS = [
    (20, 15),   # ToxicGranule vs SegNeutrophil   (D-RISE diff=0.083)
    (0,  15),   # BandNeutrophil vs SegNeutrophil  (D-RISE diff=0.121)
    (9,   2),   # Myelocyte vs Blast               (D-RISE diff=0.117)
    (8,   7),   # Monocyte vs Lymphocyte           (D-RISE diff=0.123)
    (24, 15),   # hyperSeg. vs SegNeutrophil       (D-RISE diff=0.256)
    (0,  24),   # BandNeutrophil vs hyperSeg.      (feature proximity, no D-RISE cases)
    (14, 12),   # Schistocyte vs RBC               (D-RISE diff=0.357)
    (18, 12),   # TargetCell vs RBC                (D-RISE diff=0.221)
    (3,  12),   # Echinocyte vs RBC                (D-RISE diff=0.276)
    (4,  12),   # Elliptocyte vs RBC               (D-RISE diff=0.170)
]
# PLT 쌍 제외: YOLO FPN 거리가 이미 충분히 큼

print(f'Hard negative pairs: {len(CORRECTED_HARD_PAIRS)}쌍', flush=True)
for a, b in CORRECTED_HARD_PAIRS:
    print(f'  {CLASS_NAMES[a]} <-> {CLASS_NAMES[b]}', flush=True)


class CropDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform, max_per_class=300):
        img_dir = Path(img_dir); lbl_dir = Path(lbl_dir)
        self.items = []
        counts = [0]*N_CLS
        for lbl_f in sorted(lbl_dir.glob('*.txt')):
            img_f = img_dir / (lbl_f.stem + '.jpg')
            if not img_f.exists():
                img_f = img_dir / (lbl_f.stem + '.png')
            if not img_f.exists(): continue
            img = Image.open(img_f).convert('RGB')
            W, H = img.size
            for line in lbl_f.read_text().strip().splitlines():
                if not line.strip(): continue
                p = line.split(); cls = int(p[0])
                if cls >= N_CLS: continue
                if counts[cls] >= max_per_class: continue
                cx,cy,w,h = float(p[1]),float(p[2]),float(p[3]),float(p[4])
                x1=max(0,int((cx-w/2)*W)); y1=max(0,int((cy-h/2)*H))
                x2=min(W,int((cx+w/2)*W)); y2=min(H,int((cy+h/2)*H))
                if x2-x1<8 or y2-y1<8: continue
                self.items.append((str(img_f), (x1,y1,x2,y2), cls))
                counts[cls] += 1
        self.transform = transform
        print(f'Dataset: {len(self.items)} crops', flush=True)
        for i,n in enumerate(CLASS_NAMES):
            print(f'  [{i:2d}] {n}: {counts[i]}', flush=True)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, box, cls = self.items[idx]
        img = Image.open(path).convert('RGB').crop(box)
        v1 = self.transform(img)
        v2 = self.transform(img)
        return v1, v2, cls


class EvalDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform, max_per_class=100):
        img_dir = Path(img_dir); lbl_dir = Path(lbl_dir)
        self.items = []
        counts = [0]*N_CLS
        for lbl_f in sorted(lbl_dir.glob('*.txt')):
            img_f = img_dir / (lbl_f.stem + '.jpg')
            if not img_f.exists():
                img_f = img_dir / (lbl_f.stem + '.png')
            if not img_f.exists(): continue
            img = Image.open(img_f).convert('RGB')
            W, H = img.size
            for line in lbl_f.read_text().strip().splitlines():
                if not line.strip(): continue
                p = line.split(); cls = int(p[0])
                if cls >= N_CLS: continue
                if counts[cls] >= max_per_class: continue
                cx,cy,w,h = float(p[1]),float(p[2]),float(p[3]),float(p[4])
                x1=max(0,int((cx-w/2)*W)); y1=max(0,int((cy-h/2)*H))
                x2=min(W,int((cx+w/2)*W)); y2=min(H,int((cy+h/2)*H))
                if x2-x1<8 or y2-y1<8: continue
                self.items.append((str(img_f), (x1,y1,x2,y2), cls))
                counts[cls] += 1
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, box, cls = self.items[idx]
        img = Image.open(path).convert('RGB').crop(box)
        return self.transform(img), cls


augment = T.Compose([
    T.Resize((72,72)),
    T.RandomCrop(64),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_tf = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

print('Train 데이터 로드...', flush=True)
train_ds = CropDataset('/home/smile/work/pbs06/train/images',
                       '/home/smile/work/pbs06/train/labels', augment, 300)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=(DEVICE=='cuda'), drop_last=True)

print('Val 데이터 로드...', flush=True)
val_ds = EvalDataset('/home/smile/work/pbs06/val/images',
                     '/home/smile/work/pbs06/val/labels', eval_tf, 100)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)


class SupConNet(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B,2048,1,1)
        self.proj = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
    def forward(self, x):
        h = self.backbone(x).flatten(1)  # (B,2048)
        z = F.normalize(self.proj(h), dim=1)  # (B,128)
        return h, z


class SupConLossHN(nn.Module):
    """SupCon Loss with proper hard negative weighting in forward()."""
    def __init__(self, temperature=0.07, hard_pairs=None, hard_weight=2.0):
        super().__init__()
        self.T = temperature
        self.hard_weight = hard_weight
        # Build symmetric set
        self.hard_pairs = set()
        if hard_pairs:
            for a, b in hard_pairs:
                self.hard_pairs.add((a, b))
                self.hard_pairs.add((b, a))
        print(f'SupConLossHN: T={temperature}, hard_pairs={len(self.hard_pairs)//2}쌍, hw={hard_weight}', flush=True)

    def forward(self, features, labels):
        # features: (2B, 128) L2-normalized, labels: (B,)
        B = labels.shape[0]
        labels_rep = labels.repeat(2)   # (2B,)
        dev = features.device

        # Positive pair mask (same class, excluding self)
        pos_mask = (labels_rep.unsqueeze(0) == labels_rep.unsqueeze(1)).float()
        eye = torch.eye(2*B, device=dev)
        pos_mask = pos_mask - eye  # remove self

        # Similarity
        sim = torch.matmul(features, features.T) / self.T  # (2B,2B)
        sim_self_masked = sim - eye * 1e9  # exclude self from denominator

        # Build hard negative weight matrix for denominator
        # w[i,j] = hard_weight if (label_i, label_j) is a hard pair, else 1.0
        if self.hard_pairs:
            li = labels_rep.unsqueeze(1)  # (2B,1)
            lj = labels_rep.unsqueeze(0)  # (1,2B)
            # Vectorized: for each (a,b) pair, mark (li==a)&(lj==b)
            hn_mask = torch.zeros(2*B, 2*B, device=dev)
            for a, b in self.hard_pairs:
                a_t = torch.tensor(a, device=dev)
                b_t = torch.tensor(b, device=dev)
                pair_ij = ((li == a_t) & (lj == b_t)).float()
                hn_mask = hn_mask + pair_ij
            hn_mask = hn_mask.clamp(0, 1)
            # Add log(hw) to sim for hard negative pairs → equivalent to multiplying
            # exp(sim) by hw in the denominator (standard reweighting trick)
            log_hw = torch.log(torch.tensor(self.hard_weight, device=dev))
            sim_denom = sim_self_masked + hn_mask * log_hw
        else:
            sim_denom = sim_self_masked

        # SupCon loss: for each anchor, average over positive pairs
        log_prob = sim_self_masked - torch.logsumexp(sim_denom, dim=1, keepdim=True)
        n_pos = pos_mask.sum(1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(1) / n_pos
        return loss.mean()


model = SupConNet(feat_dim=128).to(DEVICE)
total = sum(p.numel() for p in model.parameters())
print(f'파라미터: {total:,}', flush=True)

criterion = SupConLossHN(temperature=0.07, hard_pairs=CORRECTED_HARD_PAIRS, hard_weight=2.0)
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.proj.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

EPOCHS = 100
losses = []; best = float('inf')
print(f'EXP-L3b SupCon+HN 학습 시작 ({EPOCHS} epoch)', flush=True)

for epoch in range(1, EPOCHS+1):
    model.train(); ep_loss = 0.0
    for v1, v2, labels in train_loader:
        v1 = v1.to(DEVICE); v2 = v2.to(DEVICE); labels = labels.to(DEVICE)
        _, z1 = model(v1); _, z2 = model(v2)
        features = torch.cat([z1, z2], dim=0)   # (2B,128)
        loss = criterion(features, labels)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item()
    ep_loss /= len(train_loader)
    losses.append(ep_loss)
    scheduler.step()
    if ep_loss < best:
        best = ep_loss
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'loss': best,
                    'hard_pairs': list(CORRECTED_HARD_PAIRS),
                    'class_names': CLASS_NAMES},
                   OUT_DIR / 'supcon_hn_best.pt')
    if epoch % 10 == 0 or epoch == 1:
        print(f'  epoch {epoch:3d}/{EPOCHS}  loss={ep_loss:.4f}  best={best:.4f}', flush=True)

print(f'학습 완료. best={best:.4f}', flush=True)
with open(OUT_DIR / 'losses.json', 'w') as f:
    json.dump(losses, f)

# 학습 곡선
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(losses); ax.set_xlabel('epoch'); ax.set_ylabel('SupCon+HN Loss')
ax.set_title('EXP-L3b SupCon + Hard Negative Loss Curve')
plt.tight_layout(); fig.savefig(OUT_DIR / 'training_curve.png', dpi=120); plt.close()

# 특징 추출 (val set)
print('특징 추출 (val set)...', flush=True)
ckpt = torch.load(OUT_DIR / 'supcon_hn_best.pt', map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

all_feats, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        h, z = model(imgs)
        all_feats.append(h.cpu().numpy())
        all_labels.append(labels.numpy())
feats = np.concatenate(all_feats)        # (N, 2048)
labels_arr = np.concatenate(all_labels)
np.save(OUT_DIR / 'features_supcon_hn.npy', feats)
np.save(OUT_DIR / 'labels.npy', labels_arr)
print(f'특징 저장: {feats.shape}', flush=True)

# 평가
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()
X = scaler.fit_transform(feats)

print('kNN 평가...', flush=True)
results = {}
for k in [5, 10, 20]:
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(X, labels_arr)
    acc_k = accuracy_score(labels_arr, knn.predict(X))
    results[f'kNN_{k}'] = acc_k
    print(f'  kNN-{k}: {acc_k:.4f}', flush=True)

sil = silhouette_score(X, labels_arr, metric='cosine', sample_size=2000, random_state=42)
results['Silhouette'] = sil
print(f'Silhouette: {sil:.4f}', flush=True)

km = KMeans(n_clusters=25, random_state=42, n_init=10)
km.fit(X)
ari = adjusted_rand_score(labels_arr, km.labels_)
results['ARI'] = ari
print(f'ARI: {ari:.4f}', flush=True)

print('Linear Probe (5-fold CV)...', flush=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lp_accs, lp_f1s, lp_bals = [], [], []
for tr, te in skf.split(X, labels_arr):
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X[tr], labels_arr[tr])
    pred = clf.predict(X[te])
    lp_accs.append(accuracy_score(labels_arr[te], pred))
    lp_f1s.append(f1_score(labels_arr[te], pred, average='macro', zero_division=0))
    lp_bals.append(balanced_accuracy_score(labels_arr[te], pred))
results['LP_acc'] = float(np.mean(lp_accs))
results['LP_macro_f1'] = float(np.mean(lp_f1s))
results['LP_bal_acc'] = float(np.mean(lp_bals))
print(f'LP Acc={np.mean(lp_accs):.4f}  F1={np.mean(lp_f1s):.4f}  BalAcc={np.mean(lp_bals):.4f}', flush=True)

# Confusion pair distances (D-RISE 13쌍)
print('Confusion pair 거리 계산...', flush=True)
CONFUSION_PAIRS = [
    (20, 15, 'ToxicGranule↔SegNeutrophil'),
    (0,  15, 'BandNeutrophil↔SegNeutrophil'),
    (9,   2, 'Myelocyte↔Blast'),
    (8,   7, 'Monocyte↔Lymphocyte'),
    (24, 15, 'hyperSeg.↔SegNeutrophil'),
    (0,  24, 'BandNeutrophil↔hyperSeg.'),
    (14, 12, 'Schistocyte↔RBC'),
    (18, 12, 'TargetCell↔RBC'),
    (3,  12, 'Echinocyte↔RBC'),
    (4,  12, 'Elliptocyte↔RBC'),
    (13, 12, 'Reticulocyte↔RBC'),
    (19, 12, 'TearDropCell↔RBC'),
    (11,  6, 'PLT↔GiantPlt'),
]

from sklearn.preprocessing import normalize
X_norm = normalize(feats, norm='l2')  # cosine distance via L2-normalized dot product

pair_dists = {}
centroids = {}
for ci in range(N_CLS):
    mask = labels_arr == ci
    if mask.sum() > 0:
        centroids[ci] = X_norm[mask].mean(axis=0)
        centroids[ci] /= (np.linalg.norm(centroids[ci]) + 1e-9)

for a, b, name in CONFUSION_PAIRS:
    if a in centroids and b in centroids:
        cos_sim = np.dot(centroids[a], centroids[b])
        dist = 1.0 - cos_sim
        pair_dists[name] = float(dist)
        print(f'  {name}: {dist:.5f}', flush=True)
    else:
        pair_dists[name] = None
        print(f'  {name}: N/A (no samples)', flush=True)

results['confusion_pair_distances'] = pair_dists
valid_dists = [v for v in pair_dists.values() if v is not None]
results['mean_confusion_dist'] = float(np.mean(valid_dists)) if valid_dists else None
print(f'Mean confusion pair distance: {results["mean_confusion_dist"]:.4f}', flush=True)

with open(OUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('결과 저장 완료.', flush=True)

# UMAP
print('UMAP 생성 중...', flush=True)
try:
    import umap
    reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=30,
                        min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(X)
    np.save(OUT_DIR / 'umap_supcon_hn.npy', emb)
    COLORS = plt.cm.get_cmap('tab20', 25)
    fig, ax = plt.subplots(figsize=(14,10))
    for ci, name in enumerate(CLASS_NAMES):
        m = labels_arr == ci
        if m.sum() > 0:
            ax.scatter(emb[m,0], emb[m,1], c=[COLORS(ci)], label=name, s=8, alpha=0.7)
    ax.legend(fontsize=6, ncol=3, markerscale=3)
    ax.set_title('EXP-L3b SupCon+HN — UMAP (val set, ResNet-50 backbone)')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'umap_supcon_hn.png', dpi=150); plt.close()
    print('UMAP 저장 완료.', flush=True)
except Exception as e:
    print(f'UMAP 생성 실패: {e}', flush=True)

print('=== EXP-L3b 완료 ===', flush=True)
'''

ws_url = f"ws://211.232.120.231:8889/api/kernels/{kid}/channels"
ws = websocket.create_connection(ws_url, cookie=f"_xsrf={xsrf}")
msg_id = str(uuid.uuid4())
msg = {
    "header": {"msg_id": msg_id, "msg_type": "execute_request",
                "username": "", "session": "", "version": "5.0"},
    "parent_header": {}, "metadata": {},
    "content": {"code": code, "silent": False, "store_history": False,
                 "user_expressions": {}, "allow_stdin": False}
}
ws.send(json.dumps(msg))

print("EXP-L3b SupCon+HN 학습 중 (100 epoch)...")
while True:
    try: raw = ws.recv()
    except Exception as e: print("WS:", e); break
    m = json.loads(raw)
    mt = m["header"]["msg_type"]
    if mt == "stream": print(m["content"]["text"], end="", flush=True)
    elif mt == "error":
        print("ERROR:", m["content"]["ename"], m["content"]["evalue"])
        for t in m["content"]["traceback"]: print(re.sub(r'\x1b\[[0-9;]*m','',t))
        break
    elif mt == "execute_reply": print(f"\n[완료: {m['content']['status']}]"); break

ws.close()
s.delete(f"{SERVER}/api/kernels/{kid}", headers=headers)
