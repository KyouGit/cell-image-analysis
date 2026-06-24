"""EXP-005c: EigenCAM category comparison with contamination filter"""
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

WEIGHTS = "/home/smile/work/pbs/model/yolov5/runsbackup_v10/250418_U2L_001_v11/blood_cell/weights/best.pt"
OUT_BASE = "/home/smile/work/hdd8t/EXP-005c_EigenCAM"
label_dir = "/home/smile/work/pbs06/test/labels"
img_dir   = "/home/smile/work/pbs06/test/images"

CLASSES = ["BandNeutrophil","Basophil","Blast","Echinocyte","Elliptocyte",
           "Eosinophil","GiantPlt","Lymphocyte","Monocyte","Myelocyte",
           "Nucleated","PLT","RBC","Reticulocyte","Schistocyte","SegNeutrophil",
           "Smudge","Stomatocyte","TargetCell","TearDropCell","ToxicGranule",
           "ToxicVacuole","clump-PLT","degeneWBC","hyperSeg."]

WBC_CLS = {0,1,2,5,7,8,9,10,15,16,20,21,23,24}
PLT_CLS = {6,11,22}

PAIRS = [
    (14, 12, "Schistocyte_vs_RBC",            "RBC", WBC_CLS | PLT_CLS),
    (18, 12, "TargetCell_vs_RBC",             "RBC", WBC_CLS | PLT_CLS),
    (3,  12, "Echinocyte_vs_RBC",             "RBC", WBC_CLS | PLT_CLS),
    (4,  12, "Elliptocyte_vs_RBC",            "RBC", WBC_CLS | PLT_CLS),
    (19, 12, "TearDropCell_vs_RBC",           "RBC", WBC_CLS | PLT_CLS),
    (0,  15, "BandNeutrophil_vs_SegNeutrophil","WBC", PLT_CLS),
    (9,  2,  "Myelocyte_vs_Blast",            "WBC", PLT_CLS),
    (8,  7,  "Monocyte_vs_Lymphocyte",        "WBC", PLT_CLS),
    (20, 15, "ToxicGranule_vs_SegNeutrophil", "WBC", PLT_CLS),
    (6,  11, "GiantPlt_vs_PLT",              "PLT", WBC_CLS),
    (22, 11, "clumpPLT_vs_PLT",              "PLT", WBC_CLS),
]

N_SAMPLES  = 3
IMG_SIZE   = 640
PANEL_W    = 320
PANEL_H    = 320
PAD_FACTOR = 2.5

print("Loading model...", flush=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, flush=True)
model = YOLO(WEIGHTS)
yolo_model = model.model.eval().to(DEVICE)

target_layer = None
for name, module in yolo_model.named_modules():
    if name == "model.23.cv3.0.1.1.conv":
        target_layer = module
        print("EigenCAM target:", name, flush=True)
        break
if target_layer is None:
    for name, module in yolo_model.named_modules():
        if "cv3" in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
            print("Fallback target:", name, flush=True)
            break

def get_samples(class_id, exclude_classes, max_samples=N_SAMPLES):
    samples = []
    for img_path in sorted(Path(img_dir).glob("*.jpg"))[:5000]:
        lp = Path(label_dir) / (img_path.stem + ".txt")
        if not lp.exists():
            continue
        lines = lp.read_text().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) < 5 or int(parts[0]) != class_id:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            if bw * bh < 0.001:
                continue
            with Image.open(img_path) as im:
                W, H = im.size
            ch = max(bw * W, bh * H) * PAD_FACTOR / 2
            x1, y1 = cx*W - ch, cy*H - ch
            x2, y2 = cx*W + ch, cy*H + ch
            dirty = False
            if exclude_classes:
                for l in lines:
                    p = l.split()
                    if len(p) < 5 or int(p[0]) not in exclude_classes:
                        continue
                    lx, ly = float(p[1])*W, float(p[2])*H
                    if x1 <= lx <= x2 and y1 <= ly <= y2:
                        dirty = True
                        break
            if not dirty:
                samples.append((str(img_path), cx, cy, bw, bh))
                if len(samples) >= max_samples:
                    return samples
                break
    return samples

def prepare_crop(img_path, cx, cy, bw, bh):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    cx_px, cy_px = cx*W, cy*H
    bw_px, bh_px = bw*W, bh*H
    ch = max(bw_px, bh_px) * PAD_FACTOR / 2
    x1 = max(0, int(cx_px - ch));  y1 = max(0, int(cy_px - ch))
    x2 = min(W, int(cx_px + ch));  y2 = min(H, int(cy_px + ch))
    crop = img.crop((x1, y1, x2, y2))
    cW, cH = crop.size
    scale = min(IMG_SIZE/cW, IMG_SIZE/cH)
    nw, nh = int(cW*scale), int(cH*scale)
    pad = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (114,114,114))
    dx, dy = (IMG_SIZE-nw)//2, (IMG_SIZE-nh)//2
    pad.paste(crop.resize((nw, nh), Image.Resampling.LANCZOS), (dx, dy))
    buf = pad.tobytes()
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    t = t.reshape(IMG_SIZE, IMG_SIZE, 3).permute(2,0,1).float()/255.0
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    inp  = (t - mean) / std
    rgb_t = t.permute(1,2,0)
    bcx = ((cx_px-x1)*scale + dx) / IMG_SIZE
    bcy = ((cy_px-y1)*scale + dy) / IMG_SIZE
    bw2 = bw_px*scale / IMG_SIZE
    bh2 = bh_px*scale / IMG_SIZE
    return inp, rgb_t, bcx, bcy, bw2, bh2

def tensor_to_pil(t):
    u8 = (t.clamp(0,1)*255).byte()
    h, w = u8.shape[:2]
    return Image.frombytes("RGB",(w,h), bytes(u8.contiguous().reshape(-1).tolist()))

def apply_jet(c):
    c = c.clamp(0,1)
    r = (1.5-(c*4-3).abs()).clamp(0,1)
    g = (1.5-(c*4-2).abs()).clamp(0,1)
    b = (1.5-(c*4-1).abs()).clamp(0,1)
    return torch.stack([r,g,b], dim=-1)

def overlay(rgb, cam, alpha=0.55):
    return tensor_to_pil((1-alpha)*rgb + alpha*apply_jet(cam))

def compute_eigencam(inp, class_idx=None):
    inp = inp.unsqueeze(0).to(DEVICE)
    acts = {}
    h = target_layer.register_forward_hook(lambda m,i,o: acts.__setitem__("t", o.detach()))
    with torch.no_grad():
        out = yolo_model(inp)
    h.remove()
    act = acts["t"].squeeze(0).float()
    C, H, W = act.shape
    if class_idx is not None:
        si = 4 + class_idx
        preds = ([out] if isinstance(out, torch.Tensor) and out.dim()>=3 and out.shape[1]>si
                 else [p for p in out if isinstance(p,torch.Tensor) and p.dim()>=3 and p.shape[1]>si]
                 if isinstance(out,(list,tuple)) else [])
        if preds and preds[0][:,si,:].max().item() > 0:
            w = act.reshape(C,-1).mean(1).clamp(min=0)
            act = act * (w/(w.sum()+1e-8)).view(C,1,1)
    af = act.reshape(C, H*W)
    af = af - af.mean(1, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(af, full_matrices=False)
        cam_flat = Vh[0]
    except:
        cam_flat = af.mean(0)
    cam = cam_flat.reshape(H, W)
    if cam.min() < 0: cam = cam - cam.min()
    cm = cam.max()
    cam = cam/cm if cm>0 else torch.zeros_like(cam)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0).cpu(),
                        size=(IMG_SIZE,IMG_SIZE), mode="bilinear", align_corners=False)
    return cam.squeeze()

def cam_stats(cam_a, cam_b, bcx, bcy, bw, bh):
    H, W = cam_a.shape
    x1 = max(0, int((bcx-bw/2)*W)); y1 = max(0, int((bcy-bh/2)*H))
    x2 = min(W, int((bcx+bw/2)*W)); y2 = min(H, int((bcy+bh/2)*H))
    mask = torch.zeros(H, W)
    mask[y1:y2, x1:x2] = 1
    cr_a = float((cam_a*mask).sum()/(cam_a.sum()+1e-8))
    cr_b = float((cam_b*mask).sum()/(cam_b.sum()+1e-8))
    diff  = float((cam_a-cam_b).abs().mean())
    ta = cam_a >= torch.quantile(cam_a, 0.8)
    tb = cam_b >= torch.quantile(cam_b, 0.8)
    overlap = float((ta&tb).float().sum() / (ta|tb).float().sum().clamp(min=1))
    return {"center_ratio_a": cr_a, "center_ratio_b": cr_b,
            "diff_mean": diff, "peak_overlap": overlap}

def make_panel(img, title, w=PANEL_W, h=PANEL_H):
    c = Image.new("RGB",(w,h),(30,30,30))
    c.paste(img.resize((w,h-20), Image.Resampling.LANCZOS),(0,20))
    ImageDraw.Draw(c).text((4,3), title, fill=(220,220,220))
    return c

def draw_box(img, bcx, bcy, bw, bh):
    iw, ih = img.size
    d = ImageDraw.Draw(img)
    d.rectangle([int((bcx-bw/2)*iw), int((bcy-bh/2)*ih),
                 int((bcx+bw/2)*iw), int((bcy+bh/2)*ih)],
                outline=(0,255,0), width=2)

all_stats = {}

for cls_a, cls_b, pair_name, category, exclude in PAIRS:
    print(f"\n[{category}] {pair_name}", flush=True)
    out_dir = f"{OUT_BASE}/{category}"
    os.makedirs(out_dir, exist_ok=True)

    samples = get_samples(cls_a, exclude, N_SAMPLES)
    print(f"  clean samples: {len(samples)}", flush=True)
    if not samples:
        print("  skip (no clean samples)", flush=True)
        continue

    rows = []
    pair_stats = []

    for idx, (img_path, cx, cy, bw, bh) in enumerate(samples):
        inp, rgb, bcx2, bcy2, bw2, bh2 = prepare_crop(img_path, cx, cy, bw, bh)
        p0 = tensor_to_pil(rgb)
        draw_box(p0, bcx2, bcy2, bw2, bh2)
        p0 = make_panel(p0, f"[{idx+1}] {CLASSES[cls_a][:14]}")
        cam_a = compute_eigencam(inp.clone(), cls_a)
        cam_b = compute_eigencam(inp.clone(), cls_b)
        p1 = make_panel(overlay(rgb, cam_a), f"Eigen->{CLASSES[cls_a][:12]}")
        p2 = make_panel(overlay(rgb, cam_b), f"Eigen->{CLASSES[cls_b][:12]}")
        diff = (cam_a-cam_b).abs()
        diff = diff/diff.max() if diff.max()>0 else diff
        p3 = make_panel(overlay(rgb, diff, 0.6), "Diff(A-B)")
        row = Image.new("RGB",(PANEL_W*4, PANEL_H))
        for i,p in enumerate([p0,p1,p2,p3]):
            row.paste(p,(i*PANEL_W,0))
        rows.append(row)
        s = cam_stats(cam_a, cam_b, bcx2, bcy2, bw2, bh2)
        pair_stats.append(s)
        print(f"    [{idx+1}] center_ratio={s['center_ratio_a']:.2f}/{s['center_ratio_b']:.2f} "
              f"diff={s['diff_mean']:.3f} overlap={s['peak_overlap']:.2f}", flush=True)

    TH = 30
    final = Image.new("RGB",(PANEL_W*4, PANEL_H*len(rows)+TH),(20,20,20))
    ImageDraw.Draw(final).text((10,8), f"[{category}] {pair_name}", fill=(255,255,100))
    for i,r in enumerate(rows):
        final.paste(r,(0, TH+i*PANEL_H))
    save_path = f"{out_dir}/eigencam_{pair_name}.png"
    final.save(save_path, dpi=(120,120))
    print(f"  -> {save_path}", flush=True)

    def avg(key): return sum(s[key] for s in pair_stats)/len(pair_stats)
    all_stats[pair_name] = {
        "category": category,
        "n_samples": len(pair_stats),
        "center_ratio_a_mean": avg("center_ratio_a"),
        "center_ratio_b_mean": avg("center_ratio_b"),
        "diff_mean_mean": avg("diff_mean"),
        "peak_overlap_mean": avg("peak_overlap"),
        "samples": pair_stats,
    }

stats_path = f"{OUT_BASE}/stats.json"
with open(stats_path, "w") as f:
    json.dump(all_stats, f, indent=2)
print(f"\nStats saved: {stats_path}", flush=True)
print("Done.", flush=True)
