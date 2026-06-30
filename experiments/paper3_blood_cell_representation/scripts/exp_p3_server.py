"""
run_p3_v2.py
Paper 3 experiments via Jupyter API with correct file paths.
"""

import requests
import json
import time
import websocket
import uuid
import base64
import os
import sys

SERVER = "http://211.232.120.231:8889"
LOCAL_OUT_RAW = r"C:\Users\PC\Desktop\Codex\논문\03_국제저널_표현분석\실험결과"
LOCAL_OUT = LOCAL_OUT_RAW

os.makedirs(LOCAL_OUT, exist_ok=True)

s = requests.Session()
resp = s.get(SERVER, timeout=10)
print(f"Server: {resp.status_code}")
xsrf = s.cookies.get("_xsrf", "")
headers = {"X-XSRFToken": xsrf, "Content-Type": "application/json"}

r = s.post(f"{SERVER}/api/kernels", headers=headers, json={"name": "python3"}, timeout=15)
r.raise_for_status()
kid = r.json()["id"]
print(f"Kernel: {kid}")

WS_URL = f"ws://211.232.120.231:8889/api/kernels/{kid}/channels"

def run_code(code, timeout=600, label=""):
    ws = websocket.create_connection(WS_URL, timeout=timeout,
                                     header={"Cookie": "; ".join(f"{k}={v}" for k, v in s.cookies.items())})
    msg_id = str(uuid.uuid4())
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "user", "session": str(uuid.uuid4()),
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True,
                    "user_expressions": {}, "allow_stdin": False},
        "channel": "shell",
    }))
    outputs = []
    errors = []
    start = time.time()
    if label:
        print(f"\n[{label}]")
    while time.time() - start < timeout:
        try:
            ws.settimeout(10)
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        msg = json.loads(raw)
        mt = msg.get("header", {}).get("msg_type", "")
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        if mt == "stream":
            t = msg["content"].get("text", "")
            outputs.append(t)
            print(t, end="", flush=True)
        elif mt == "execute_result":
            t = msg["content"].get("data", {}).get("text/plain", "")
            outputs.append(t + "\n")
            print(t)
        elif mt == "error":
            errors.append(msg["content"].get("ename", "") + ": " + msg["content"].get("evalue", ""))
            print("ERR:", errors[-1])
        elif mt == "execute_reply":
            break
    ws.close()
    return "".join(outputs), errors

# ─── Step 1: Setup and load data ─────────────────────────────────────────────
SETUP_CODE = """
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os
warnings.filterwarnings('ignore')

CLASS_NAMES = ['BandNeutrophil','Basophil','Blast','Echinocyte','Elliptocyte',
               'Eosinophil','GiantPlt','Lymphocyte','Monocyte','Myelocyte',
               'Nucleated','PLT','RBC','Reticulocyte','Schistocyte',
               'SegNeutrophil','Smudge','Stomatocyte','TargetCell','TearDropCell',
               'ToxicGranule','ToxicVacuole','clump-PLT','degeneWBC','hyperSeg.']

WBC_IDS = [0,1,2,5,7,8,9,10,15,16,20,21,23,24]
RBC_IDS = [3,4,12,13,14,17,18,19]
PLT_IDS = [6,11,22]

feat_yolo = np.load('/home/smile/work/hdd8t/EXP-L1_FoundationFeatures/features_yolo_fpn.npy')
feat_dino = np.load('/home/smile/work/hdd8t/EXP-L1_FoundationFeatures/features_dinov2.npy')
lbl_yolo  = np.load('/home/smile/work/hdd8t/EXP-L1_FoundationFeatures/labels.npy')
feat_sc   = np.load('/home/smile/work/hdd8t/EXP-L3_SupCon/features_supcon.npy')
lbl_sc    = np.load('/home/smile/work/hdd8t/EXP-L3_SupCon/labels.npy')

def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n == 0, 1e-12, n)
    return X / n

feat_yolo_n = l2norm(feat_yolo.astype(np.float32))
feat_dino_n = l2norm(feat_dino.astype(np.float32))
feat_sc_n   = l2norm(feat_sc.astype(np.float32))

def get_centroids(feat_n, labels, n_classes=25):
    cents = np.zeros((n_classes, feat_n.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            cents[c] = feat_n[mask].mean(axis=0)
    return l2norm(cents)

c_yolo = get_centroids(feat_yolo_n, lbl_yolo)
c_dino = get_centroids(feat_dino_n, lbl_yolo)
c_sc   = get_centroids(feat_sc_n,   lbl_sc)

print(f"YOLO: {feat_yolo.shape}, DINOv2: {feat_dino.shape}, SupCon: {feat_sc.shape}")
print(f"Centroids — YOLO: {c_yolo.shape}, DINOv2: {c_dino.shape}, SupCon: {c_sc.shape}")
print("Data loaded OK")
"""
run_code(SETUP_CODE, label="SETUP")

# ─── Step 2: EXP-P3-1 All 300 pair distances ─────────────────────────────────
EXP1_CODE = """
pairs = list(combinations(range(25), 2))
records = []
for i, j in pairs:
    d_y = float(1 - np.dot(c_yolo[i], c_yolo[j]))
    d_d = float(1 - np.dot(c_dino[i], c_dino[j]))
    d_s = float(1 - np.dot(c_sc[i],   c_sc[j]))
    records.append({'id_i': i, 'id_j': j,
                    'class_i': CLASS_NAMES[i], 'class_j': CLASS_NAMES[j],
                    'dist_yolo': round(d_y, 6),
                    'dist_dinov2': round(d_d, 6),
                    'dist_supcon': round(d_s, 6)})

df_pairs = pd.DataFrame(records)
df_pairs.to_csv('/tmp/all_pair_distances.csv', index=False)
print(f"all_pair_distances.csv: {len(df_pairs)} pairs")
print(f"YOLO mean={df_pairs.dist_yolo.mean():.4f}, DINOv2 mean={df_pairs.dist_dinov2.mean():.4f}, SupCon mean={df_pairs.dist_supcon.mean():.4f}")
print("Top 10 smallest YOLO distances (most confused):")
print(df_pairs.nsmallest(10, 'dist_yolo')[['class_i','class_j','dist_yolo','dist_supcon']].to_string())
"""
run_code(EXP1_CODE, label="EXP-P3-1")

# ─── Step 3: EXP-P3-2 Per-class compactness vs AP50 ──────────────────────────
EXP2_CODE = """
AP50 = {
    'BandNeutrophil':0.710,'Basophil':0.971,'Blast':0.943,'Echinocyte':0.872,
    'Elliptocyte':0.917,'Eosinophil':0.986,'GiantPlt':0.898,'Lymphocyte':0.937,
    'Monocyte':0.946,'Myelocyte':0.888,'Nucleated':0.901,'PLT':0.961,'RBC':0.975,
    'Reticulocyte':0.890,'Schistocyte':0.679,'SegNeutrophil':0.916,'Smudge':0.973,
    'Stomatocyte':0.929,'TargetCell':0.901,'TearDropCell':0.828,'ToxicGranule':0.882,
    'ToxicVacuole':0.880,'clump-PLT':0.900,'degeneWBC':0.927,'hyperSeg.':0.764
}

compact_records = []
for c_idx in range(25):
    cname = CLASS_NAMES[c_idx]
    mask = lbl_yolo == c_idx
    n = int(mask.sum())
    if n < 2:
        compact_records.append({'class_id':c_idx,'class_name':cname,'n_samples':n,
                                 'mean_dist_to_centroid':None,'std_dist':None,'ap50':AP50.get(cname)})
        continue
    feats = feat_yolo_n[mask]
    centroid = c_yolo[c_idx]
    dists = 1 - feats @ centroid
    compact_records.append({'class_id':c_idx,'class_name':cname,'n_samples':n,
                             'mean_dist_to_centroid':round(float(np.mean(dists)),6),
                             'std_dist':round(float(np.std(dists)),6),
                             'ap50':AP50.get(cname)})

df_compact = pd.DataFrame(compact_records)
df_compact.to_csv('/tmp/per_class_compactness.csv', index=False)
valid = df_compact.dropna()
r_p, p_p = pearsonr(valid['mean_dist_to_centroid'], valid['ap50'])
r_s, p_s = spearmanr(valid['mean_dist_to_centroid'], valid['ap50'])
print(f"per_class_compactness.csv saved ({len(df_compact)} rows)")
print(f"Pearson r={r_p:.4f} p={p_p:.4f}")
print(f"Spearman r={r_s:.4f} p={p_s:.4f}")
print(valid[['class_name','mean_dist_to_centroid','ap50']].sort_values('mean_dist_to_centroid').to_string())
"""
run_code(EXP2_CODE, label="EXP-P3-2")

# ─── Step 4: Generate all figures ────────────────────────────────────────────
FIGS_CODE = """
import matplotlib.cm as cm
import numpy as np

AP50 = {
    'BandNeutrophil':0.710,'Basophil':0.971,'Blast':0.943,'Echinocyte':0.872,
    'Elliptocyte':0.917,'Eosinophil':0.986,'GiantPlt':0.898,'Lymphocyte':0.937,
    'Monocyte':0.946,'Myelocyte':0.888,'Nucleated':0.901,'PLT':0.961,'RBC':0.975,
    'Reticulocyte':0.890,'Schistocyte':0.679,'SegNeutrophil':0.916,'Smudge':0.973,
    'Stomatocyte':0.929,'TargetCell':0.901,'TearDropCell':0.828,'ToxicGranule':0.882,
    'ToxicVacuole':0.880,'clump-PLT':0.900,'degeneWBC':0.927,'hyperSeg.':0.764
}

valid = df_compact.dropna()
r_p, p_p = pearsonr(valid['mean_dist_to_centroid'], valid['ap50'])

# ─ Figure 1: feature_error_correlation.png ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
colors_by_cat = []
for _, row in valid.iterrows():
    cid = int(row['class_id'])
    if cid in WBC_IDS:   colors_by_cat.append('#E63946')
    elif cid in RBC_IDS: colors_by_cat.append('#457B9D')
    else:                colors_by_cat.append('#2A9D8F')

ax.scatter(valid['mean_dist_to_centroid'], valid['ap50'],
           c=colors_by_cat, s=110, alpha=0.85, edgecolors='white', linewidth=0.7, zorder=5)

for _, row in valid.iterrows():
    if row['ap50'] < 0.85 or row['mean_dist_to_centroid'] > 0.12:
        ax.annotate(row['class_name'], (row['mean_dist_to_centroid'], row['ap50']),
                    fontsize=8, ha='left', va='bottom',
                    xytext=(4, 3), textcoords='offset points')

z = np.polyfit(valid['mean_dist_to_centroid'], valid['ap50'], 1)
p_line = np.poly1d(z)
xr = np.linspace(valid['mean_dist_to_centroid'].min(), valid['mean_dist_to_centroid'].max(), 100)
ax.plot(xr, p_line(xr), 'k--', alpha=0.4, linewidth=1.5, label=f'Linear fit (r={r_p:.3f})')

legend_handles = [
    mpatches.Patch(color='#E63946', label='WBC'),
    mpatches.Patch(color='#457B9D', label='RBC'),
    mpatches.Patch(color='#2A9D8F', label='PLT'),
]
ax.legend(handles=legend_handles + [plt.Line2D([0],[0],linestyle='--',color='k',alpha=0.4,label=f'Trend r={r_p:.3f}')], fontsize=10)
ax.set_xlabel('Mean Cosine Distance to Class Centroid (YOLO FPN)\\n← more compact    more spread →', fontsize=11)
ax.set_ylabel('AP50 (Paper 1 YOLO detection)', fontsize=11)
ax.set_title(f'Per-class Feature Compactness vs Detection AP50\\n(Pearson r={r_p:.3f}, p={p_p:.3f})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/feature_error_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_error_correlation.png")

# ─ Figure 2: cross_model_comparison.png ─────────────────────────────────────
DRISE_PAIRS_WBC = [
    ((20,15), 'ToxicGranule\\nvs SegNeu.'),
    ((9,2),   'Myelocyte\\nvs Blast'),
    ((0,15),  'BandNeu.\\nvs SegNeu.'),
    ((8,7),   'Monocyte\\nvs Lympho.'),
    ((24,15), 'hyperSeg.\\nvs SegNeu.'),
    ((0,24),  'BandNeu.\\nvs hyperSeg.'),
]
DRISE_PAIRS_RBC = [
    ((4,12),  'Elliptocyte\\nvs RBC'),
    ((18,12), 'TargetCell\\nvs RBC'),
    ((3,12),  'Echinocyte\\nvs RBC'),
    ((19,12), 'TearDrop\\nvs RBC'),
    ((14,12), 'Schistocyte\\nvs RBC'),
]
DRISE_PAIRS_PLT = [
    ((6,11),  'GiantPlt\\nvs PLT'),
    ((22,11), 'clump-PLT\\nvs PLT'),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_colors = {'YOLO FPN':'#F4A261', 'DINOv2':'#E9C46A', 'SupCon':'#2A9D8F'}
bar_w = 0.22

for ax, (cat_name, cat_pairs) in zip(axes, [('WBC',DRISE_PAIRS_WBC),('RBC',DRISE_PAIRS_RBC),('PLT',DRISE_PAIRS_PLT)]):
    x_pos = np.arange(len(cat_pairs))
    for offset, (model_name, c_mat) in zip([-0.25,0.0,0.25], [('YOLO FPN',c_yolo),('DINOv2',c_dino),('SupCon',c_sc)]):
        dists = [float(1 - np.dot(c_mat[p[0][0]], c_mat[p[0][1]])) for p in cat_pairs]
        bars = ax.bar(x_pos + offset, dists, bar_w, color=model_colors[model_name],
                      label=model_name, edgecolor='white', linewidth=0.5)
        for bar, d in zip(bars, dists):
            if d > 0.02:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f'{d:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p[1] for p in cat_pairs], fontsize=9)
    ax.set_title(f'{cat_name} Confusion Pairs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Distance' if ax is axes[0] else '')
    ax.set_ylim(0, 0.90)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle('Cross-Model Distance: YOLO FPN vs DINOv2 vs SupCon (D-RISE confusion pairs)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/cross_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved cross_model_comparison.png")

print("Figures done.")
"""
run_code(FIGS_CODE, label="FIGURES")

# ─── Step 5: 4-panel story figure (UMAP needed) ───────────────────────────────
UMAP_CODE = """
import umap
import matplotlib.cm as cm

np.random.seed(42)

def subsample(feat, labels, max_per_class=120):
    idx = []
    for c in range(25):
        ci = np.where(labels == c)[0]
        if len(ci) > max_per_class:
            ci = np.random.choice(ci, max_per_class, replace=False)
        idx.extend(ci)
    idx = np.array(idx)
    return feat[idx], labels[idx]

fy_s, ly_s = subsample(feat_yolo_n, lbl_yolo)
fs_s, ls_s = subsample(feat_sc_n,   lbl_sc)
print(f"UMAP subsamples: YOLO {fy_s.shape}, SupCon {fs_s.shape}")

print("Running UMAP YOLO...")
emb_y = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(fy_s)
print("Running UMAP SupCon...")
emb_s = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(fs_s)
print("UMAP done.")

# ─ 4-panel figure ─────────────────────────────────────────────────────────────
cmap25 = cm.get_cmap('tab20', 25)

DRISE_PAIRS_ALL = [
    ((20,15),'ToxicGranule/SegNeu.'),((9,2),'Myelocyte/Blast'),
    ((0,15),'BandNeu./SegNeu.'),((8,7),'Monocyte/Lympho.'),
    ((24,15),'hyperSeg./SegNeu.'),((0,24),'BandNeu./hyperSeg.'),
    ((4,12),'Elliptocyte/RBC'),((18,12),'TargetCell/RBC'),
    ((3,12),'Echinocyte/RBC'),((19,12),'TearDrop/RBC'),
    ((14,12),'Schistocyte/RBC'),((6,11),'GiantPlt/PLT'),((22,11),'clump-PLT/PLT'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.patch.set_facecolor('#F8F9FA')
axes = axes.flatten()

# Panel A: YOLO UMAP
ax = axes[0]
for c in range(25):
    mask = ly_s == c
    if mask.sum() > 0:
        ax.scatter(emb_y[mask,0], emb_y[mask,1], s=10, alpha=0.55,
                   color=cmap25(c), label=CLASS_NAMES[c] if mask.sum() > 0 else '')
ax.set_title('(A) YOLO FPN Feature Space\\nSilhouette=0.038', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP-1', fontsize=10); ax.set_ylabel('UMAP-2', fontsize=10)
ax.tick_params(labelsize=9)

# Panel B: distance vs confusion
ax = axes[1]
AP50 = {
    'BandNeutrophil':0.710,'Basophil':0.971,'Blast':0.943,'Echinocyte':0.872,
    'Elliptocyte':0.917,'Eosinophil':0.986,'GiantPlt':0.898,'Lymphocyte':0.937,
    'Monocyte':0.946,'Myelocyte':0.888,'Nucleated':0.901,'PLT':0.961,'RBC':0.975,
    'Reticulocyte':0.890,'Schistocyte':0.679,'SegNeutrophil':0.916,'Smudge':0.973,
    'Stomatocyte':0.929,'TargetCell':0.901,'TearDropCell':0.828,'ToxicGranule':0.882,
    'ToxicVacuole':0.880,'clump-PLT':0.900,'degeneWBC':0.927,'hyperSeg.':0.764
}
valid2 = df_compact.dropna()
colors_b = ['#E63946' if int(r['class_id']) in WBC_IDS else '#457B9D' if int(r['class_id']) in RBC_IDS else '#2A9D8F' for _,r in valid2.iterrows()]
ax.scatter(valid2['mean_dist_to_centroid'], valid2['ap50'], c=colors_b, s=80, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
for _, row in valid2.iterrows():
    if row['ap50'] < 0.85 or row['mean_dist_to_centroid'] > 0.11:
        ax.annotate(str(row['class_name'])[:10], (row['mean_dist_to_centroid'], row['ap50']),
                    fontsize=7, xytext=(3,2), textcoords='offset points')
z2 = np.polyfit(valid2['mean_dist_to_centroid'], valid2['ap50'], 1)
xr = np.linspace(valid2['mean_dist_to_centroid'].min(), valid2['mean_dist_to_centroid'].max(), 100)
ax.plot(xr, np.poly1d(z2)(xr), 'k--', alpha=0.4, linewidth=1.5)
r_pb, _ = pearsonr(valid2['mean_dist_to_centroid'], valid2['ap50'])
ax.set_xlabel('Within-class Spread (YOLO FPN)', fontsize=10)
ax.set_ylabel('AP50', fontsize=10)
ax.set_title(f'(B) Feature Compactness vs Detection AP50\\nPearson r={r_pb:.3f}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
legend_h = [mpatches.Patch(color='#E63946',label='WBC'),mpatches.Patch(color='#457B9D',label='RBC'),mpatches.Patch(color='#2A9D8F',label='PLT')]
ax.legend(handles=legend_h, fontsize=9)

# Panel C: SupCon UMAP
ax = axes[2]
for c in range(25):
    mask = ls_s == c
    if mask.sum() > 0:
        ax.scatter(emb_s[mask,0], emb_s[mask,1], s=10, alpha=0.55,
                   color=cmap25(c), label=CLASS_NAMES[c])
ax.set_title('(C) SupCon Feature Space\\nSilhouette=0.500', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP-1', fontsize=10); ax.set_ylabel('UMAP-2', fontsize=10)
ax.tick_params(labelsize=9)

# Shared legend for panels A and C
handles = [mpatches.Patch(color=cmap25(c), label=CLASS_NAMES[c]) for c in range(25)]
fig.legend(handles=handles, loc='upper right', ncol=2, fontsize=7,
           bbox_to_anchor=(1.01, 1), title='Classes', title_fontsize=8)

# Panel D: 13-pair bar comparison
ax = axes[3]
y_idxs = np.arange(len(DRISE_PAIRS_ALL))
d_y = [float(1 - np.dot(c_yolo[p[0][0]], c_yolo[p[0][1]])) for p in DRISE_PAIRS_ALL]
d_d = [float(1 - np.dot(c_dino[p[0][0]], c_dino[p[0][1]])) for p in DRISE_PAIRS_ALL]
d_s = [float(1 - np.dot(c_sc[p[0][0]],   c_sc[p[0][1]])) for p in DRISE_PAIRS_ALL]
short_lbl = [p[1] for p in DRISE_PAIRS_ALL]

ax.barh(y_idxs - 0.27, d_y, 0.24, color='#F4A261', label='YOLO FPN', alpha=0.9)
ax.barh(y_idxs,         d_d, 0.24, color='#E9C46A', label='DINOv2', alpha=0.9)
ax.barh(y_idxs + 0.27, d_s, 0.24, color='#2A9D8F', label='SupCon', alpha=0.9)
ax.set_yticks(y_idxs)
ax.set_yticklabels(short_lbl, fontsize=8.5)
ax.set_xlabel('Cosine Distance', fontsize=10)
ax.set_title('(D) Confusion Pair Distances (13 D-RISE pairs)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, axis='x', alpha=0.3)
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=5.5, color='#E63946', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=10.5, color='#457B9D', linestyle='--', alpha=0.3, linewidth=1)
for yp, cat_lbl in [(2.7,'WBC'),(8,'RBC'),(11.5,'PLT')]:
    ax.text(0.72, yp, cat_lbl, ha='center', va='center', fontsize=9, color='gray', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

fig.suptitle('Representation Space Geometry Predicts Blood Cell Detection Confusion\\n'
             'YOLO FPN: near-zero separation → SupCon: 44.5× increase across 13 confusion pairs',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/story_figure_4panel.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved story_figure_4panel.png")

# Verify all files
for f in ['/tmp/all_pair_distances.csv','/tmp/per_class_compactness.csv',
          '/tmp/feature_error_correlation.png','/tmp/cross_model_comparison.png',
          '/tmp/story_figure_4panel.png']:
    sz = os.path.getsize(f) if os.path.exists(f) else 0
    print(f"  {os.path.basename(f)}: {sz} bytes")
print("ALL DONE")
"""
run_code(UMAP_CODE, label="UMAP+STORY_FIG", timeout=600)

# ─── Step 6: Download all results ────────────────────────────────────────────
ENCODE_CODE = """
import base64, json

files = ['/tmp/all_pair_distances.csv','/tmp/per_class_compactness.csv',
         '/tmp/feature_error_correlation.png','/tmp/cross_model_comparison.png',
         '/tmp/story_figure_4panel.png']
encoded = {}
for f in files:
    if os.path.exists(f):
        with open(f,'rb') as fh:
            encoded[os.path.basename(f)] = base64.b64encode(fh.read()).decode()
        print(f"Encoded {os.path.basename(f)}")
with open('/tmp/p3_encoded.json','w') as fh:
    json.dump(encoded, fh)
print(f"Keys: {list(encoded.keys())}")
"""
run_code(ENCODE_CODE, label="ENCODE")

DOWNLOAD_CODE = """
import json
with open('/tmp/p3_encoded.json') as f:
    data = json.load(f)
for k, v in data.items():
    print(f"FILE_START:{k}")
    print(v)
    print(f"FILE_END:{k}")
print("TRANSFER_COMPLETE")
"""
print("\n[DOWNLOAD]")
dl_stdout, _ = run_code(DOWNLOAD_CODE, label="DOWNLOAD", timeout=120)

# Parse and save
lines = dl_stdout.split('\n')
i = 0
saved = []
while i < len(lines):
    line = lines[i]
    if line.startswith('FILE_START:'):
        fname = line[len('FILE_START:'):]
        b64_chunks = []
        i += 1
        while i < len(lines) and not lines[i].startswith('FILE_END:'):
            b64_chunks.append(lines[i])
            i += 1
        try:
            file_bytes = base64.b64decode(''.join(b64_chunks))
            out_path = os.path.join(LOCAL_OUT, fname)
            with open(out_path, 'wb') as f:
                f.write(file_bytes)
            saved.append(out_path)
            print(f"Saved: {fname} ({len(file_bytes)} bytes)")
        except Exception as e:
            print(f"FAILED {fname}: {e}")
    i += 1

print(f"\nTotal downloaded: {len(saved)} files")
for f in saved:
    print(f"  {f}")

try:
    s.delete(f"{SERVER}/api/kernels/{kid}", headers=headers, timeout=10)
    print("Kernel cleaned up.")
except:
    pass
