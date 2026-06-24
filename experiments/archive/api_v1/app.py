import io
import json
import os
import cv2
import base64
import uuid
import copy
from flask_cors import CORS

from yolov5 import detect
import shutil

from flask import Flask, jsonify, request

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins':['http://localhost:3000', 'https://labeler.u2cloud.co.kr']}})

WORKING_DIR = "/home/smile/work/pbs02-blood-cell-detection/"
INFIMAGE_DIR = os.path.join(WORKING_DIR, 'infimage')
UPLOAD_DIR = os.path.join(WORKING_DIR, 'upload')
YOLO_DIR = os.path.join(WORKING_DIR, 'yolov5')

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'This is a flask application'})

@app.route('/inf', methods=['POST'])
def post_inf():
    params = request.get_json()
    fileInfo = convertBase64ToImageFile(params['memberId'], params['imageId'], params['image'], INFIMAGE_DIR);
    result = inf(fileInfo[0], fileInfo[1], params['width'], params['height'])
    return result

@app.route('/upload', methods=['POST'])
def post_upload():
    params = request.get_json()
    fileInfo = convertBase64ToImageFile(params['memberId'], params['imageId'], params['image'], UPLOAD_DIR);
    annotations = params['annotations'];
    
    isFileExists = os.path.isfile('annotations.csv');
    
    f = open('annotations.csv', 'a');
    
    if not isFileExists:
        f.write('image,xmin,ymin,xmax,ymax,label\r\n');
    
    for annotation in annotations:
        annotation_dict = json.loads(annotation['boundingBox']);
        xmin = annotation_dict['x'];
        ymin = annotation_dict['y'];
        xmax = annotation_dict['x'] + annotation_dict['width'];
        ymax = annotation_dict['y'] + annotation_dict['height'];
        print(fileInfo[0] + '.' + fileInfo[1] + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + annotation['label'] + '\r\n');
        f.write(fileInfo[0] + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + annotation['label'] + '\r\n');
        
    f.close();
    
    return 'OK'

def convertBase64ToImageFile(memberId, imageId, base64string, folder):
    if not (os.path.exists(folder)):
        os.mkdir(folder)
        
    len_full = len(base64string)
    len_header = base64string.find(',')
    
    header = base64string[0 : base64string.find(';')]
    imageFormat = header[header.find('/') + 1:]
    
    imgdata = base64.b64decode(base64string[len_header + 1 - len_full:])
    
    fileid = f'{memberId}-{imageId}-{uuid.uuid4()}'
    filename = os.path.join(folder, f'{fileid}.{imageFormat}')
    
    with open(filename, 'wb') as f:
        f.write(imgdata)
        
    return fileid, imageFormat

def inf(fileid, imageFormat, width, height):
    weights = os.path.join(YOLO_DIR, 'runsbackup/70img100ephoc/blood_cell/weights/best.pt')
    inf_project = os.path.join(INFIMAGE_DIR)
    inf_name = fileid
    inf_path = os.path.join(inf_project, inf_name)
    inf_target_filename = fileid
    inf_target_fileext = imageFormat
    inf_target_folder = os.path.join(INFIMAGE_DIR)
    result_file = os.path.join(inf_path, 'labels', inf_target_filename) + '.txt';
    
    print(result_file)
    
    source = os.path.join(inf_target_folder, inf_target_filename + '.' + inf_target_fileext)

    if (os.path.exists(inf_path )):
        shutil.rmtree(inf_path)

    detect.run(
        weights = weights
        , source = source
        , name = inf_name
        , project = inf_project
        , save_txt = True
        , save_conf = True
        , save_crop = True    
        , imgsz = (width, height)
        , max_det = 10000
    )
    
    result_file_stream = open(result_file)
    lines = result_file_stream.readlines();

    titles = ['class', 'x1', 'y1', 'x2', 'y2', 'prob']
    tmpdict = {}
    result_json = []

    for line in lines:
        tmp = line.split(' ');
        for i in range(len(tmp)):
            tmpdict[titles[i]] = tmp[i].replace('\n', '')
            
        result_json.append(copy.deepcopy(tmpdict))
    
    return json.dumps(result_json)

if __name__ == '__main__':
    app.run()