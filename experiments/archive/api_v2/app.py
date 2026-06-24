import io
import json
import os
import base64
import uuid
import copy
import shutil

from flask_cors import CORS
from flask import Flask, jsonify, request

import u2_utils
import u2_imagesplitter

# Yolo V5 소스 경로
WORKING_DIR = "/home/smile/work/pbs/model"
# Yolo V5 Weight 경로
MODEL_DIR = "/home/smile/work/pbs/model/yolov5/runsbackup"

# U2Labeler 소스 업로드 경로
DATASET_DIR = "/home/smile/work/pbs/dataset/u2labeler"
# 인퍼런스용 이미지 업로드 경로
INFIMAGE_DIR = "/home/smile/work/pbs/infimage"

app = Flask(__name__)
# CORS(app, resources={r'/*': {'origins':'http://localhost:3000'}})
# CORS(app, resources={r'/*': {'origins':['http://localhost:3000', 'https://labeler.u2cloud.co.kr']}})


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'This is a flask application'})

####################################################################################################
#
# 인퍼런스
# 지정된 모델을 이용하여 하나의 이미지 내에 존재하는 셀을 찾아냄
# 
# PARAMETERS
#    image : base64로 인코딩된 이미지
#    width : 이미지 너비
#    height : 이미지 높이
#    model_name : 인퍼런스시 사용할 모델 이름 (get_models 함수를 이용해 얻은 이름이어야 함)
#
####################################################################################################
@app.route('/inf', methods=['POST'])
def post_inf():
    params = request.get_json();
    
    fileName = u2_utils.save_image_file(INFIMAGE_DIR, uuid.uuid4(), params['image']);
    result = u2_utils.inf(params['model_name'], fileName, params['width'], params['height'], MODEL_DIR, INFIMAGE_DIR)
    print(result)
    return result

####################################################################################################
#
# 서버에 저장된 모델 목록 가져오기
# post_inf 함수의 model_name 파라미터에 들어갈 모델 이름은 이 API 를 통해 가져온 모델 목록에 속한 이름이어야 함
# 
####################################################################################################
@app.route('/get_models', methods=['GET'])
def get_models():
    dir_name = WORKING_DIR
    weights = []
    weight = {}

    for model_name in os.listdir(dir_name):
        if not model_name.startswith('.'):
            for weight_name in os.listdir(os.path.join(dir_name, model_name, 'runsbackup')):
                if not weight_name.startswith('.'):
                    weight['model_name'] = model_name
                    weight['weight_name'] = weight_name
                    weight['fullpath'] = os.path.join(dir_name, model_name, 'runsbackup', weight_name)
                    weights.append(copy.deepcopy(weight))
    
    return json.dumps(weights)

####################################################################################################
#
# 이미지 내 셀들의 BoundingBox 추출
# 
# PARAMETERS
#    image : base64로 인코딩된 이미지
#
####################################################################################################
@app.route('/aa', methods=['POST'])
def aa():
    params = request.get_json();
    
    fileName = u2_utils.save_image_file(INFIMAGE_DIR, uuid.uuid4(), params['image']);
    result = u2_utils.inf(params['model_name'], fileName, params['width'], params['height'], MODEL_DIR, INFIMAGE_DIR)
    print(result)
    return result


@app.route('/uploadprepare', methods=['POST'])
def post_uploadprepare():
    params = request.get_json();
    folder = os.path.join(DATASET_DIR, str(params['projectId']), str(params['sourceImageSetId']));
    
    # 업로드 대상 폴더 존재시 삭제 후 재생성
    if (os.path.exists(folder)):
        shutil.rmtree(folder);   
        
    os.makedirs(folder, exist_ok=True);
    
    # 어노테이션 파일 생성
    annotationFileName = os.path.join(folder, str(params['sourceImageSetId']) + '.csv');
    
    f = open(annotationFileName, 'w');
    f.write('image,xmin,ymin,xmax,ymax,label\r\n');
    f.close();
    
    return 'OK'                       
    
@app.route('/upload', methods=['POST'])
def post_upload():
    params = request.get_json();
    folder = os.path.join(DATASET_DIR, str(params['projectId']), str(params['sourceImageSetId']));
    imageFileName = u2_utils.save_image_file(folder, params['imageId'], params['image']);
    u2_utils.saveAnnotation(folder, imageFileName, params['projectId'], params['sourceImageSetId'], params['annotations'])
    
    return 'OK'

####################################################################################################
#
# 원시 데이터 업로드 관련 함수
#
####################################################################################################

####################################################################################################
#
# 폴더를 삭제
# 원시 이미지를 업로드 하기 전 이미지가 업로드 될 폴더를 삭제 함
# 
# PARAMETERS
#    folder : 삭제할 폴더
#
####################################################################################################
@app.route('/source_folder_delete', methods=['POST'])
def source_folder_delete():
    params = request.get_json();
    
    if os.path.exists(params['folder']):
        shutil.rmtree(params['folder'])
    return 'OK'
    
####################################################################################################
#
# 원시 이미지 업로드
# 
# PARAMETERS
#    folder : 저장할 폴더
#    image_filename : 이미지 파일 이름 (파일이름만, 확장자 없을 경우 이미지 포맷에 맞추어 생성)
#    image : 이미지에 대한 base64로 인코딩 된 문자열
#    annotation_filename : 어노테이션 파일 이름
#    annotation : 어노테이션 (JSON)
#
# EXAMPLE
#    save_image_file('/home/smile/work/pbs/api/tmp/', '12345.jpg', 'data:image/jpeg;base64, ...')
#
####################################################################################################
@app.route('/source_upload', methods=['POST'])
def source_upload():
    params = request.get_json();
    
    u2_utils.save_image_file(params['folder'], params['image_filename'], params['image'])   
    u2_utils.save_annotation_file(params['folder'], params['annotation_filename'], params['annotation'])
    return 'OK'

####################################################################################################
#
# 이미지를 특정 크기로 자르기
# RBC 의 경우 학습하기에는 이미지가 크고 인스턴스가 많아 224 X 224 크기로 잘라야 함
# RBC 이미지 업로드 후 이 함수를 호출하여 이미지를 자름
#
####################################################################################################
@app.route('/source_split', methods=['POST'])
def source_split():
    params = request.get_json();
    
    image_filename = os.path.join(params['folder'], params['image_filename'])
    annotation_filename = os.path.join(params['folder'], params['annotation_filename'])
    split_annotation_filename = os.path.join(params['folder'], 'split.csv')
    delete_original_data = params['delete_original_data']

    annotations_new = pd.DataFrame(columns=['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'source', 'orderId', 'classType'])
    IMAGE_CROP_SIZE = 224

    u2_imagesplitter.SplitImage(image_filename, annotation_filename, annotations_new, IMAGE_CROP_SIZE)
    annotations_new.to_csv(split_annotation_filename, index=False)
    
    if delete_original_data == True:
        os.remove(image_filename)
        os.remove(annotation_filename)
    
    return 'OK'

####################################################################################################
#
# find_boudingbox
# 이미지 내 셀들의 바운딩 박스 추출
#
# PARAMETERS
#    params['image'] : base64 인코딩 된 바운딩박스를 추출할 이미지 파일
#
# RETURN
#    바운딩 박스 리스트
#    {
#        "result_code": "OK", 
#        "result": [
#            {"x1": 77, "y1": 218, "x2": 110, "y2": 256}, 
#            {"x1": 165, "y1": 210, "x2": 224, "y2": 256}
#        ]
#    }
#
####################################################################################################
@app.route('/find_boundingbox', methods=['POST'])
def find_boundingbox():
    params = request.get_json();
    
    fileName = u2_utils.save_image_file(INFIMAGE_DIR, uuid.uuid4(), params['image']);

    bounding_boxes = u2_utils.find_boudingbox(fileName, 20)

    result_json = {'result_code':'OK', 'result':bounding_boxes}

    return json.dumps(result_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)
    