"""서버에서 실행: Schistocyte 10개 후보 EigenCAM 생성 후 저장"""
import os, json, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ultralytics import YOLO

WEIGHTS   = "/home/smile/work/pbs/model/yolov5/runsbackup_v10/250418_U2L_001_v11/blood_cell/weights/best.pt"
img_dir   = "/home/smile/work/pbs06/test/images"
OUT_DIR   = "/home/smile/work/hdd8t/schisto_cam_candidates"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 640
PAD_FACTOR = 2.0
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

CANDIDATES = [
    {"idx":0,"file":"1324885.jpg","cx":0.266304,"cy":0.314171,"bw":0.163043,"bh":0.200535},
    {"idx":1,"file":"1780731.jpg","cx":0.252717,"cy":0.754011,"bw":0.168478,"bh":0.219251},
    {"idx":2,"file":"1818852.jpg","cx":0.763587,"cy":0.406417,"bw":0.13587,"bh":0.197861},
    {"idx":3,"file":"1884953.jpg","cx":0.574728,"cy":0.276738,"bw":0.1875,"bh":0.136364},
    {"idx":4,"file":"1946283.jpg","cx":0.525815,"cy":0.755348,"bw":0.1875,"bh":0.21123},
    {"idx":5,"file":"2007515.jpg","cx":0.30163,"cy":0.302139,"bw":0.201087,"bh":0.176471},
    {"idx":6,"file":"2009271.jpg","cx":0.620924,"cy":0.537433,"bw":0.19837,"bh":0.187166},
    {"idx":7,"file":"2137516.jpg","cx":0.32337,"cy":0.229947,"bw":0.163043,"bh":0.187166},
    {"idx":8,"file":"2165449.jpg","cx":0.527174,"cy":0.258021,"bw":0.152174,"bh":0.173797},
    {"idx":9,"file":"2165589.jpg","cx":0.305707,"cy":0.453209,"bw":0.138587,"bh":0.184492},
]

print("Loading model...", flush=True)
model = YOLO(WEIGHTS)
yolo_model = model.model.eval().to(DEVICE)

target_layer = None
for name, module in yolo_model.named_modules():
    if name == "model.23.cv3.0.1.1.conv":
        target_layer = module
        print("Target:", name, flush=True)
        break
if target_layer is None:
    for name, module in yolo_model.named_modules():
        if "cv3" in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
            print("Fallback target:", name, flush=True)
            break

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

def apply_jet(c):
    c = c.clamp(0,1)
    r = (1.5-(c*4-3).abs()).clamp(0,1)
    g = (1.5-(c*4-2).abs()).clamp(0,1)
    b = (1.5-(c*4-1).abs()).clamp(0,1)
    return torch.stack([r,g,b], dim=-1)

def tensor_to_pil(t):
    u8 = (t.clamp(0,1)*255).byte()
    h, w = u8.shape[:2]
    return Image.frombytes("RGB",(w,h), bytes(u8.contiguous().reshape(-1).tolist()))

def overlay(rgb, cam, alpha=0.55):
    return tensor_to_pil((1-alpha)*rgb + alpha*apply_jet(cam))

def compute_eigencam(inp, bcx, bcy, bw2, bh2):
    inp = inp.unsqueeze(0).to(DEVICE)
    acts = {}
    h = target_layer.register_forward_hook(lambda m,i,o: acts.__setitem__("t", o.detach()))
    with torch.no_grad():
        yolo_model(inp)
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

FONT = ImageFont.load_default()
PANEL = 280

for c in CANDIDATES:
    idx = c["idx"]
    fpath = os.path.join(img_dir, c["file"])
    print(f"Processing #{idx} {c['file']}...", flush=True)
    inp, rgb_t, bcx, bcy, bw2, bh2 = prepare_crop(fpath, c["cx"], c["cy"], c["bw"], c["bh"])
    cam = compute_eigencam(inp, bcx, bcy, bw2, bh2)

    # 원본 crop
    orig = tensor_to_pil(rgb_t)
    orig_r = orig.resize((PANEL, PANEL), Image.Resampling.LANCZOS)

    # CAM overlay
    cam_img = overlay(rgb_t, cam)
    cam_r = cam_img.resize((PANEL, PANEL), Image.Resampling.LANCZOS)

    # 나란히 합치기
    TH = 22
    out = Image.new("RGB", (PANEL*2+4, PANEL+TH), (30,30,30))
    out.paste(orig_r, (0, TH))
    out.paste(cam_r, (PANEL+4, TH))
    d = ImageDraw.Draw(out)
    d.text((4, 4), f"#{idx} Original ({c['file']})", fill=(200,200,200), font=FONT)
    d.text((PANEL+8, 4), f"#{idx} EigenCAM", fill=(255,180,100), font=FONT)

    out_path = f"{OUT_DIR}/schisto_cam_{idx:02d}_{Path(c['file']).stem}.png"
    out.save(out_path)
    print(f"  Saved: {out_path}", flush=True)

print("ALL DONE", flush=True)
