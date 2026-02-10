"""
U2Bio Blood Cell Analysis Utilities (Production Reference)

Core utility functions for the U2Bio blood cell labeling and analysis platform:
- Image I/O: base64 decoding, file saving
- YOLO v10 inference: object detection with bounding box output
- VAE-based OOD detection: filter out-of-distribution (blurry/invalid) images
- Annotation management: CSV-based bounding box storage
- Edge detection: Canny-based auto bounding box proposals
- Authentication: JWT token verification
- Logging: MariaDB-based action logging

Note: Server paths, DB credentials, and model weights have been removed.
      This code is provided as a reference for the production architecture.
"""

import os
import sys
import json
import copy
import base64
import shutil
import traceback
from flask import Flask, jsonify, request
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from ultralytics import YOLO
import pymysql

# Database configuration (loaded from environment variables)
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": os.getenv("DB_CHARSET", "utf8mb3")
}


# ══════════════════════════════════════════════════════════════
# Image I/O
# ══════════════════════════════════════════════════════════════

def save_image_file(folder, image_filename, base64string):
    """Decode base64 image string and save to disk"""
    os.makedirs(folder, exist_ok=True)
    f, e = os.path.splitext(str(image_filename))

    header = base64string[0:base64string.find(';')]
    imageFormat = header[header.find('/') + 1:]
    if imageFormat == 'jpeg':
        imageFormat = 'jpg'

    len_full = len(base64string)
    len_header = base64string.find(',')
    imgdata = base64.b64decode(base64string[len_header + 1 - len_full:])
    filename = os.path.join(folder, f'{f}.{imageFormat}')

    with open(filename, 'wb') as f:
        f.write(imgdata)

    # Re-save via OpenCV for format consistency
    tmpimg = cv2.imread(filename)
    cv2.imwrite(filename, tmpimg)
    return filename


def save_annotation_file(folder, annotation_filename, annotation):
    """Save annotation text to file"""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, annotation_filename), 'w') as f:
        f.write(annotation)


def saveAnnotation(folder, imageFileName, projectId, sourceImageSetId, annotations):
    """Append bounding box annotations to CSV file"""
    os.makedirs(folder, exist_ok=True)
    annotationFileName = os.path.join(folder, str(sourceImageSetId) + '.csv')

    with open(annotationFileName, 'a') as f:
        for annotation in annotations:
            annotation_dict = json.loads(annotation['boundingBox'])
            xmin = annotation_dict['x']
            ymin = annotation_dict['y']
            xmax = annotation_dict['x'] + annotation_dict['width']
            ymax = annotation_dict['y'] + annotation_dict['height']
            line = f"{imageFileName},{xmin},{ymin},{xmax},{ymax},{annotation['label']}"
            f.write(line + '\r\n')


# ══════════════════════════════════════════════════════════════
# YOLOv10 Inference
# ══════════════════════════════════════════════════════════════

def inf_v10(model_name, filename, width, height, model_dir, infimage_dir):
    """
    Run YOLOv10 object detection on a single image.

    Returns JSON with detected cells:
    {result_code: "OK", result: [{class, x1, y1, x2, y2, prob}, ...]}
    """
    try:
        weights = os.path.join(model_dir, model_name, 'blood_cell/weights/best.pt')
        label_path = os.path.join(model_dir, model_name, 'label.txt')

        model = YOLO(weights)

        with open(label_path, 'r') as label_file:
            class_labels = label_file.readline().strip().split(',')

        d, f = os.path.split(filename)
        f, e = os.path.splitext(f)
        result_dir = os.path.join('runs', f)
        os.makedirs(result_dir, exist_ok=True)

        results = model.predict(source=filename, save=False,
                                imgsz=(width, height), conf=0.75)

        img = cv2.imread(filename)
        result_json = {'result_code': 'OK', 'result': []}

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                class_name = class_labels[int(cls)]
                result_json['result'].append({
                    'class': class_name,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'prob': float(conf)
                })

        # Cleanup temporary files
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)

        return json.dumps(result_json)

    except Exception as e:
        return json.dumps({'result_code': 'ERR', 'error_trace': traceback.format_exc()})


# ══════════════════════════════════════════════════════════════
# VAE-based OOD (Out-of-Distribution) Detection
# ══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))


class AdvancedVAE_UNet(nn.Module):
    """
    U-Net style Variational AutoEncoder for OOD detection.

    Architecture:
        Encoder: 3 residual blocks with max pooling -> latent space (z_dim=128)
        Decoder: 3 residual blocks with skip connections -> reconstructed image

    OOD detection: If reconstruction error > threshold, image is likely OOD.
    Additional check: Laplacian blur score to filter blurry images.
    """
    def __init__(self, z_dim=128):
        super().__init__()
        self.enc1 = ResidualBlock(3, 32)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(32, 64)
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(64, 128)
        self.down3 = nn.MaxPool2d(2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, z_dim)
        self.fc_dec = nn.Linear(z_dim, 128 * 4 * 4)
        self.dec3 = ResidualBlock(128 + 128, 64)
        self.dec2 = ResidualBlock(64 + 64, 32)
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 32, 3, 3, padding=1), nn.Sigmoid())
        self.dropout = nn.Dropout(0.3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x1 = self.enc1(x)
        x = self.down1(x1)
        x2 = self.enc2(x)
        x = self.down2(x2)
        x3 = self.enc3(x)
        x = self.down3(x3)
        pooled = self.pool(x)
        flat = self.flatten(pooled)
        return self.fc_mu(flat), self.fc_logvar(flat), [x1, x2, x3]

    def decode(self, z, skips):
        x = self.fc_dec(z).view(-1, 128, 4, 4)
        x = F.interpolate(x, size=skips[2].shape[2:])
        x = self.dec3(self.dropout(torch.cat([x, skips[2]], dim=1)))
        x = F.interpolate(x, size=skips[1].shape[2:])
        x = self.dec2(self.dropout(torch.cat([x, skips[1]], dim=1)))
        x = F.interpolate(x, size=skips[0].shape[2:])
        return self.dec1(self.dropout(torch.cat([x, skips[0]], dim=1)))

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, skips), mu, logvar


# Note: In production, model is loaded at server startup:
# model = AdvancedVAE_UNet().cuda()
# model.load_state_dict(torch.load("<model_path>"))
# model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def reconstruction_error(x, model):
    """Compute MSE between input and VAE reconstruction"""
    model.eval()
    with torch.no_grad():
        x = x.cuda()
        x_hat, mu, logvar = model(x)
        mse = F.mse_loss(x_hat, x, reduction='none')
        err = mse.view(mse.size(0), -1).mean(dim=1)
    return err.cpu().numpy()


def laplacian_blur_score(image_path):
    """Compute Laplacian variance as blur metric (lower = blurrier)"""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_ood_image(image_path, Threshold=0.00013, Laplacian_Threshold=20):
    """
    Determine if an image is Out-of-Distribution.

    Dual criteria:
    1. VAE reconstruction error > threshold (abnormal content)
    2. Laplacian blur score < threshold (too blurry)

    Returns: (reconstruction_error, ood_flag)
    """
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        err = reconstruction_error(x, model)[0]

    laplacian_score = laplacian_blur_score(image_path)
    ood_flag = 1 if (err > Threshold or laplacian_score < Laplacian_Threshold) else 0
    return err, ood_flag


# ══════════════════════════════════════════════════════════════
# Edge Detection based BBox Proposals
# ══════════════════════════════════════════════════════════════

def find_boudingbox(fileName, minimum_size):
    """Detect bounding boxes via Canny edge detection + contour analysis"""
    img = cv2.imread(fileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > minimum_size and h > minimum_size:
            bounding_boxes.append({'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h})
    return bounding_boxes


# ══════════════════════════════════════════════════════════════
# Authentication & Logging
# ══════════════════════════════════════════════════════════════

def check_token(TEST_TOKEN_VERIFY_URL, PROD_TOKEN_VERIFY_URL):
    """Verify JWT token via external auth API"""
    headers = request.headers
    req_type = headers.get('Reqtype', 'P')
    authorization = headers.get('Authorization')
    token = headers.get('Accesstoken')

    if not authorization or not token:
        return jsonify({"error": "Missing Authorization or Access Token"}), 401

    verify_url = TEST_TOKEN_VERIFY_URL if req_type == "T" else PROD_TOKEN_VERIFY_URL

    try:
        response = requests.get(verify_url,
                                headers={"Accesstoken": token, "Authorization": authorization})
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({"error": "Invalid or expired token",
                            "code": response.status_code}), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Token verification failed: {str(e)}", "code": 500}), 500


def log_action(source_type, req_type, team_id, editor_id, log_type, log_data):
    """Log actions to MariaDB for monitoring and audit"""
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        sql = """INSERT INTO LogServer (sourceType, reqType, teamId, editorId, logType, logData, creationDate)
                 VALUES (%s, %s, %s, %s, %s, %s, NOW())"""
        cursor.execute(sql, (source_type, req_type, team_id, editor_id, log_type, log_data))
        conn.commit()
    except Exception as e:
        print(f"Logging failed: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
