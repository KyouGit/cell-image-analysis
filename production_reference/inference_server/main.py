import io
import json
import os
import base64
import uuid
import copy
import shutil

from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

origins = [
    "http://localhost"
]

class Item(BaseModel):
    model_name: str
    width: int
    height: int
    image: str

#from flask_cors import CORS
#from flask import Flask, jsonify, request

import u2_utils

INFIMAGE_DIR = "/app/yolo/infimage"
MODEL_DIR = "/app/yolo/ultralytics/runsbackup"
# MODEL_DIR = "/home/smile/ai/work/u2dlp-pbs/u2pbsai-inf/yolo/ultralytics/runsbackup"

#app = Flask(__name__)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#@app.route('/', methods=['GET'])
@app.get('/')
def root():
    return {'msg' : 'This is a fastapi application'}


####################################################################################################
#
# 서버에 저장된 모델 목록 가져오기
# post_inf 함수의 model_name 파라미터에 들어갈 모델 이름은 이 API 를 통해 가져온 모델 목록에 속한 이름이어야 함
# 
####################################################################################################
@app.get('/get_models')
def get_models():
    dir_name = MODEL_DIR
    weights = []
    weight = {}

    for weight_name in os.listdir(dir_name):
        if not weight_name.startswith('.'):
            weight['model_name'] = 'yolov10'
            weight['weight_name'] = weight_name
            weight['fullpath'] = os.path.join(dir_name, weight_name)
            weights.append(copy.deepcopy(weight))
    
    return json.dumps(weights)

@app.post('/inf')
def post_inf(item: Item):
    fileName = u2_utils.save_image_file(INFIMAGE_DIR, uuid.uuid4(), item.image)
    print(fileName)
    result = u2_utils.inf_v10(item.model_name, fileName, item.width, item.height, MODEL_DIR, INFIMAGE_DIR) 

    print(result)
    return result
