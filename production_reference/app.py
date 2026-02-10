"""
Blood Cell Analysis AI Server (Production Reference)

Flask-based REST API server for U2Bio's blood cell analysis platform.
Provides endpoints for:
- YOLOv10 cell detection inference (single & batch)
- VAE-based Out-of-Distribution (OOD) image filtering
- Image upload, annotation management, and splitting
- Model management and monitoring

Note: Server paths, credentials, and SSL certificates have been removed.
      This code is provided as a reference for the production architecture.

Original Environment: Linux (Ubuntu), Dual NVIDIA A100 GPU, MariaDB
"""

import io
import json
import os
import base64
import uuid
import copy
import shutil
import ssl
from dotenv import load_dotenv
from flask_cors import CORS
from flask import Flask, jsonify, request
import requests
import u2_utils
import u2_imagesplitter
import ultralytics
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from u2_utils import is_ood_image, check_token, log_action

load_dotenv()

# Directory configuration (loaded from .env)
WORKING_DIR = os.getenv("WORKING_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
DATASET_DIR = os.getenv("DATASET_DIR")
INFIMAGE_DIR = os.getenv("INFIMAGE_DIR")

app = Flask(__name__)

TEST_TOKEN_VERIFY_URL = os.getenv("TOKEN_VERIFY_URL_TEST")
PROD_TOKEN_VERIFY_URL = os.getenv("TOKEN_VERIFY_URL_PROD")


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg': 'This is a flask application'})


# ──────────────────────────────────────────────────────────────
# Single Image Inference: YOLO Detection + OOD Filtering
# ──────────────────────────────────────────────────────────────
@app.route('/inf', methods=['POST'])
def post_inf():
    token_result = check_token(TEST_TOKEN_VERIFY_URL, PROD_TOKEN_VERIFY_URL)
    if isinstance(token_result, tuple):
        log_action("A", request.headers.get('Reqtype', 'P'),
                   str(request.headers.get('teamId', "0")),
                   str(request.headers.get('memberId', "0")),
                   "F", f"call /inf {token_result[1]} from {request.url}/{request.remote_addr}")
        return token_result

    params = request.get_json()
    fileName = u2_utils.save_image_file(INFIMAGE_DIR, uuid.uuid4(), params['image'])

    try:
        result = u2_utils.inf_v10(params['model_name'], fileName,
                                   params['width'], params['height'],
                                   MODEL_DIR, INFIMAGE_DIR)
        # VAE-based OOD detection
        err, is_ood = is_ood_image(fileName)

        if is_ood:
            if os.path.exists(fileName):
                os.remove(fileName)
            return jsonify({"OOD failed"}), 500

        log_action("A", request.headers.get('Reqtype', 'P'),
                   str(request.headers.get('teamId', "0")),
                   str(request.headers.get('memberId', "0")),
                   "S", f"call /inf 200 from {request.url}/{request.remote_addr}")
        if os.path.exists(fileName):
            os.remove(fileName)
        return result

    except Exception as e:
        if os.path.exists(fileName):
            os.remove(fileName)
        log_action("A", request.headers.get('Reqtype', 'P'),
                   str(request.headers.get('teamId', "0")),
                   str(request.headers.get('memberId', "0")),
                   "F", f"call /inf 500 from {request.url}/{request.remote_addr} - Error: {str(e)}")
        return jsonify({"error": "Inference failed"}), 500


# ──────────────────────────────────────────────────────────────
# Batch Inference: Parallel processing with ThreadPoolExecutor
# ──────────────────────────────────────────────────────────────
@app.route('/infImages', methods=['POST'])
def post_inf_images():
    token_result = check_token(TEST_TOKEN_VERIFY_URL, PROD_TOKEN_VERIFY_URL)
    if isinstance(token_result, tuple):
        log_action("A", request.headers.get('Reqtype', 'P'),
                   str(request.headers.get('teamId', "0")),
                   str(request.headers.get('memberId', "0")),
                   "F", f"call /infImages {token_result[1]} from {request.url}/{request.remote_addr}")
        return token_result

    params = request.get_json()
    model_name = params['model_name']
    images = params['images']
    has_failure = False
    results = []

    def process_image(image_data):
        image_id = image_data['imageId']
        width = image_data['width']
        height = image_data['height']
        base64_image = image_data['image']

        file_name = u2_utils.save_image_file(INFIMAGE_DIR, uuid.uuid4(), base64_image)

        # VAE-based OOD filtering per image
        err, is_ood = is_ood_image(file_name)
        if is_ood:
            if os.path.exists(file_name):
                os.remove(file_name)
            return {"imageId": image_id, "ood": True,
                    "reconstruction_error": round(err, 5), "inf": None}

        inf_result = u2_utils.inf_v10(model_name, file_name, width, height,
                                       MODEL_DIR, INFIMAGE_DIR)
        if os.path.exists(file_name):
            os.remove(file_name)
        return {"imageId": image_id, "ood": False,
                "inf": json.loads(inf_result)['result']}

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_image = {executor.submit(process_image, img): img for img in images}
            for future in as_completed(future_to_image):
                try:
                    results.append(future.result())
                except Exception as e:
                    has_failure = True
                    log_action("A", request.headers.get('Reqtype', 'P'),
                               str(request.headers.get('teamId', "0")),
                               str(request.headers.get('memberId', "0")),
                               "E", f"Error processing image: {str(e)}")

        if not has_failure:
            log_action("A", request.headers.get('Reqtype', 'P'),
                       str(request.headers.get('teamId', "0")),
                       str(request.headers.get('memberId', "0")),
                       "S", f"call /infImages 200 from {request.url}/{request.remote_addr}")
            return jsonify({"result_code": "OK", "result": results}), 200
        else:
            return jsonify({"result_code": "PARTIAL_FAILURE", "result": results,
                            "message": "Some images failed to process"}), 207

    except Exception as e:
        log_action("A", request.headers.get('Reqtype', 'P'),
                   str(request.headers.get('teamId', "0")),
                   str(request.headers.get('memberId', "0")),
                   "F", f"call /infImages 500 - Error: {str(e)}")
        return jsonify({"error": "Inference failed for multiple images"}), 500


# ──────────────────────────────────────────────────────────────
# Model Management: List available YOLO models
# ──────────────────────────────────────────────────────────────
@app.route('/get_models', methods=['GET'])
def get_models():
    token_result = check_token(TEST_TOKEN_VERIFY_URL, PROD_TOKEN_VERIFY_URL)
    if isinstance(token_result, tuple):
        return token_result

    weights = []
    for model_name in os.listdir(WORKING_DIR):
        if not model_name.startswith('.'):
            for weight_name in os.listdir(os.path.join(WORKING_DIR, model_name, 'runsbackup_v10')):
                if not weight_name.startswith('.'):
                    weights.append({
                        'model_name': model_name,
                        'weight_name': weight_name,
                        'fullpath': os.path.join(WORKING_DIR, model_name, 'runsbackup_v10', weight_name)
                    })
    return json.dumps(weights)


# ──────────────────────────────────────────────────────────────
# Data Pipeline: Upload, Annotation, Splitting
# ──────────────────────────────────────────────────────────────
@app.route('/uploadprepare', methods=['POST'])
def post_uploadprepare():
    """Initialize upload directory and annotation CSV"""
    token_result = check_token(TEST_TOKEN_VERIFY_URL, PROD_TOKEN_VERIFY_URL)
    if isinstance(token_result, tuple):
        return token_result

    params = request.get_json()
    folder = os.path.join(DATASET_DIR, str(params['projectId']), str(params['sourceImageSetId']))

    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        annotationFileName = os.path.join(folder, str(params['sourceImageSetId']) + '.csv')
        with open(annotationFileName, 'w') as f:
            f.write('image,xmin,ymin,xmax,ymax,label\r\n')
        return 'OK'
    except Exception as e:
        return jsonify({"error": "Failed to prepare upload"}), 500


@app.route('/upload', methods=['POST'])
def post_upload():
    """Upload image with bounding box annotations"""
    token_result = check_token(TEST_TOKEN_VERIFY_URL, PROD_TOKEN_VERIFY_URL)
    if isinstance(token_result, tuple):
        return token_result

    params = request.get_json()
    folder = os.path.join(DATASET_DIR, str(params['projectId']), str(params['sourceImageSetId']))

    try:
        imageFileName = u2_utils.save_image_file(folder, params['imageId'], params['image'])
        u2_utils.saveAnnotation(folder, imageFileName, params['projectId'],
                                 params['sourceImageSetId'], params['annotations'])
        return 'OK'
    except Exception as e:
        return jsonify({"error": "Failed to upload"}), 500


@app.route('/source_split', methods=['POST'])
def source_split():
    """Split high-resolution images into 224x224 tiles with annotation mapping"""
    token_result = check_token(TEST_TOKEN_VERIFY_URL, PROD_TOKEN_VERIFY_URL)
    if isinstance(token_result, tuple):
        return token_result

    params = request.get_json()
    image_filename = os.path.join(params['folder'], params['image_filename'])
    annotation_filename = os.path.join(params['folder'], params['annotation_filename'])
    split_annotation_filename = os.path.join(params['folder'], 'split.csv')
    IMAGE_CROP_SIZE = 224

    try:
        import pandas as pd
        annotations_new = pd.DataFrame(
            columns=['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'source', 'orderId', 'classType'])
        u2_imagesplitter.SplitImage(image_filename, annotation_filename, annotations_new, IMAGE_CROP_SIZE)
        annotations_new.to_csv(split_annotation_filename, index=False)

        if params.get('delete_original_data'):
            os.remove(image_filename)
            os.remove(annotation_filename)
        return 'OK'
    except Exception as e:
        return jsonify({"error": "Failed to split source"}), 500


@app.route('/find_boundingbox', methods=['POST'])
def find_boundingbox():
    """Detect bounding boxes using edge detection (Canny)"""
    token_result = check_token(TEST_TOKEN_VERIFY_URL, PROD_TOKEN_VERIFY_URL)
    if isinstance(token_result, tuple):
        return token_result

    params = request.get_json()
    try:
        fileName = u2_utils.save_image_file(INFIMAGE_DIR, uuid.uuid4(), params['image'])
        bounding_boxes = u2_utils.find_boudingbox(fileName, 20)
        return json.dumps({'result_code': 'OK', 'result': bounding_boxes})
    except Exception as e:
        return jsonify({"error": "Failed to find bounding boxes"}), 500


if __name__ == '__main__':
    # SSL certificates configured via environment
    cert_path = os.getenv("SSL_CERT_PATH")
    key_path = os.getenv("SSL_KEY_PATH")
    app.run(host='0.0.0.0', port=8000, debug=True, ssl_context=(cert_path, key_path))
