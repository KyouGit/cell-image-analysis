
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

WEIGHTS   = "/home/smile/work/pbs/model/yolov5/runsbackup_v10/250418_U2L_001_v11/blood_cell/weights/best.pt"
OUT_DIR   = "/home/smile/work/hdd8t/EXP-005c_EigenCAM_v7/RBC"
label_dir = "/home/smile/work/pbs06/test/labels"
img_dir   = "/home/smile/work/pbs06/test/images"

CLASSES = ["BandNeutrophil","Basophil","Blast","Echinocyte","Elliptocyte",
           "Eosinophil","GiantPlt","Lymphocyte","Monocyte","Myelocyte",
           "Nucleated","PLT","RBC","Reticulocyte","Schistocyte","SegNeutrophil",
           "Smudge","Stomatocyte","TargetCell","TearDropCell","ToxicGranule",
           "ToxicVacuole","clump-PLT","degeneWBC","hyperSeg."]

WBC_CLS = {0,1,2,5,7,8,9,10,15,16,20,21,23,24}
PLT_CLS = {6,11,22}
exclude = WBC_CLS | PLT_CLS

N_SAMPLES  = 3
IMG_SIZE   = 640
PANEL_W    = 280
PANEL_H    = 280
PAD_FACTOR = 2.0
CROP_MARGIN = 4

print("Loading model...", flush=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO(WEIGHTS)
yolo_model = model.model.eval().to(DEVICE)

target_layer = None
for name, module in yolo_model.named_modules():
    if name == "model.23.cv3.0.1.1.conv":
        target_layer = module; break
if target_layer is None:
    for name, module in yolo_model.named_modules():
        if "cv3" in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module; break

FONT = ImageFont.load_default()

def is_crop_clean(img_path, cx, cy, bw, bh):
    with Image.open(img_path) as im:
        W, H = im.size
        ch = max(bw*W, bh*H) * PAD_FACTOR / 2
        x1 = max(0, int(cx*W - ch)); y1 = max(0, int(cy*H - ch))
        x2 = min(W, int(cx*W + ch)); y2 = min(H, int(cy*H + ch))
        crop = np.array(im.crop((x1, y1, x2, y2)).convert("L"))
    if crop.min() < 20: return False
    if (crop > 235).mean() > 0.20: return False
    if crop.std() < 8: return False
    return True

def get_samples(class_id, excl, max_samples=N_SAMPLES, skip_first=0):
    samples = []
    skipped = 0
    for img_path in sorted(Path(img_dir).glob("*.jpg"))[:8000]:
        lp = Path(label_dir) / (img_path.stem + ".txt")
        if not lp.exists(): continue
        lines = lp.read_text().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) < 5 or int(parts[0]) != class_id: continue
            cx, cy, bw, bh = map(float, parts[1:5])
            if bw * bh < 0.001: continue
            with Image.open(img_path) as im: W, H = im.size
            ch = max(bw*W, bh*H) * PAD_FACTOR / 2
            x1, y1 = cx*W - ch, cy*H - ch
            x2, y2 = cx*W + ch, cy*H + ch
            if x1 < CROP_MARGIN or y1 < CROP_MARGIN or x2 > W-CROP_MARGIN or y2 > H-CROP_MARGIN:
                continue
            if not is_crop_clean(str(img_path), cx, cy, bw, bh): continue
            dirty = False
            for l in lines:
                p = l.split()
                if len(p) < 5 or int(p[0]) not in excl: continue
                lx, ly = float(p[1])*W, float(p[2])*H
                if x1 <= lx <= x2 and y1 <= ly <= y2:
                    dirty = True; break
            if not dirty:
                if skipped < skip_first:
                    skipped += 1
                    break
                samples.append((str(img_path), cx, cy, bw, bh))
                if len(samples) >= max_samples: return samples
                break
    return samples

def prepare_crop(img_path, cx, cy, bw, bh):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    cx_px, cy_px = cx*W, cy*H
    bw_px, bh_px = bw*W, bh*H
    ch = max(bw_px, bh_px) * PAD_FACTOR / 2
    x1 = max(0, int(cx_px - ch)); y1 = max(0, int(cy_px - ch))
    x2 = min(W, int(cx_px + ch)); y2 = min(H, int(cy_px + ch))
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

def compute_eigencam(inp, bcx, bcy, bw2, bh2):
    inp = inp.unsqueeze(0).to(DEVICE)
    acts = {}
    h = target_layer.register_forward_hook(lambda m,i,o: acts.__setitem__("t", o.detach()))
    with torch.no_grad(): yolo_model(inp)
    h.remove()
    act = acts["t"].squeeze(0).float()
    C, fH, fW = act.shape
    af = act.reshape(C, fH*fW)
    af = af - af.mean(1, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(af, full_matrices=False)
        cam_flat = Vh[0]
    except:
        cam_flat = af.mean(0)
    cam = cam_flat.reshape(fH, fW)
    bx1 = max(0, int((bcx - bw2/2) * fW))
    bx2 = min(fW, int((bcx + bw2/2) * fW))
    by1 = max(0, int((bcy - bh2/2) * fH))
    by2 = min(fH, int((bcy + bh2/2) * fH))
    if bx2 > bx1 and by2 > by1:
        if cam[by1:by2, bx1:bx2].mean() < cam.mean():
            cam = -cam
    if cam.min() < 0: cam = cam - cam.min()
    cm = cam.max()
    cam = cam/cm if cm > 0 else torch.zeros_like(cam)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0).cpu(),
                        size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return cam.squeeze()

def put_text(img, xy, text, fill=(220,220,220)):
    ImageDraw.Draw(img).text(xy, text[:35], fill=fill, font=FONT)

def make_panel(img, title, w=PANEL_W, h=PANEL_H, title_color=(220,220,220)):
    TH = 18
    c = Image.new("RGB", (w, h), (30,30,30))
    c.paste(img.resize((w, h-TH), Image.Resampling.LANCZOS), (0, TH))
    put_text(c, (4, 3), title, fill=title_color)
    return c

def draw_box(img, bcx, bcy, bw, bh, color=(255,220,0)):
    iw, ih = img.size
    ImageDraw.Draw(img).rectangle(
        [int((bcx-bw/2)*iw), int((bcy-bh/2)*ih),
         int((bcx+bw/2)*iw), int((bcy+bh/2)*ih)],
        outline=color, width=2)

DIVIDER_W = 4

# Schistocyte: skip_first=3 (기존 3개 건너뜀)
# RBC: skip_first=15 (앞의 5 pairs × 3 samples = 15개 건너뜀)
print("Getting Schistocyte samples (skip_first=3)...", flush=True)
samples_a = get_samples(14, exclude, N_SAMPLES, skip_first=3)
print(f"  Schistocyte: {len(samples_a)}", flush=True)

print("Getting RBC samples (skip_first=15)...", flush=True)
samples_b = get_samples(12, exclude, N_SAMPLES, skip_first=15)
print(f"  RBC: {len(samples_b)}", flush=True)

n = min(len(samples_a), len(samples_b), N_SAMPLES)
if n == 0:
    print("ERROR: no valid samples", flush=True)
    exit(1)

ROW_W = PANEL_W * 4 + DIVIDER_W
TH_TITLE = 28
HDR_H = 18
rows = []

for i in range(n):
    inp_a, rgb_a, bcx_a, bcy_a, bw_a, bh_a = prepare_crop(*samples_a[i])
    inp_b, rgb_b, bcx_b, bcy_b, bw_b, bh_b = prepare_crop(*samples_b[i])
    cam_a = compute_eigencam(inp_a, bcx_a, bcy_a, bw_a, bh_a)
    cam_b = compute_eigencam(inp_b, bcx_b, bcy_b, bw_b, bh_b)

    p0 = tensor_to_pil(rgb_a)
    draw_box(p0, bcx_a, bcy_a, bw_a, bh_a, color=(255,220,0))
    p0 = make_panel(p0, f"[{i+1}] Schistocyte", title_color=(255,230,100))
    p1 = make_panel(overlay(rgb_a, cam_a), "EigenCAM-Schistocyte", title_color=(255,180,80))
    p2 = tensor_to_pil(rgb_b)
    draw_box(p2, bcx_b, bcy_b, bw_b, bh_b, color=(100,200,255))
    p2 = make_panel(p2, f"[{i+1}] RBC", title_color=(130,200,255))
    p3 = make_panel(overlay(rgb_b, cam_b), "EigenCAM-RBC", title_color=(80,180,255))

    row = Image.new("RGB", (ROW_W, PANEL_H), (30,30,30))
    row.paste(p0, (0, 0))
    row.paste(p1, (PANEL_W, 0))
    for dx in range(DIVIDER_W):
        for dy in range(PANEL_H):
            row.putpixel((PANEL_W*2 + dx, dy), (80,80,80))
    row.paste(p2, (PANEL_W*2 + DIVIDER_W, 0))
    row.paste(p3, (PANEL_W*3 + DIVIDER_W, 0))
    rows.append(row)
    print(f"  [{i+1}] done", flush=True)

final_h = PANEL_H * len(rows) + TH_TITLE + HDR_H
final = Image.new("RGB", (ROW_W, final_h), (20,20,20))
d = ImageDraw.Draw(final)
put_text(final, (10, 8), "[RBC] Schistocyte_vs_RBC EigenCAM Comparison", fill=(255,255,100))
col_labels = [
    (0,                    "Schistocyte orig"),
    (PANEL_W,              "Schistocyte EigenCAM"),
    (PANEL_W*2+DIVIDER_W,  "RBC orig"),
    (PANEL_W*3+DIVIDER_W,  "RBC EigenCAM"),
]
for xpos, lbl in col_labels:
    d.rectangle([xpos, TH_TITLE, xpos+PANEL_W-1, TH_TITLE+HDR_H], fill=(45,45,45))
    put_text(final, (xpos+4, TH_TITLE+3), lbl, fill=(190,190,190))
for i, r in enumerate(rows):
    final.paste(r, (0, TH_TITLE+HDR_H+i*PANEL_H))

save_path = f"{OUT_DIR}/eigencam_Schistocyte_vs_RBC.png"
final.save(save_path, dpi=(120, 120))
print(f"  -> {save_path}", flush=True)
print("Done.", flush=True)
