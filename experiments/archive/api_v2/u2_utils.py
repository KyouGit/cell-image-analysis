import os
import cv2
import base64
import shutil
import json
import sys
import traceback
import copy

# Yolo V5 소스 경로
WORKING_DIR = "/home/smile/work/pbs/model"
sys.path.append(WORKING_DIR)
from yolov5 import detect

####################################################################################################
#
# 이미지를 파일로 저장
# 
# PARAMETERS
#    folder : 저장할 폴더
#    image_filename : 이미지 파일 이름 (파일이름만, 확장자 없을 경우 이미지 포맷에 맞추어 생성)
#    base64string : 이미지에 대한 base64로 인코딩 된 문자열
#
# EXAMPLE
#    save_image_file('/home/smile/work/pbs/api/tmp/', '12345.jpg', 'data:image/jpeg;base64, ...')
#
####################################################################################################
def save_image_file(folder, image_filename, base64string):
    os.makedirs(folder, exist_ok=True);
    
    f, e = os.path.splitext(str(image_filename))
    
    # 이미지 저장
    len_full = len(base64string)
    len_header = base64string.find(',')
    
    header = base64string[0 : base64string.find(';')]
    imageFormat = header[header.find('/') + 1:]
    if imageFormat == 'jpeg':
        imageFormat = 'jpg'
    
    imgdata = base64.b64decode(base64string[len_header + 1 - len_full:])
    
    filename = os.path.join(folder, f'{f}.{imageFormat}')
    
    with open(filename, 'wb') as f:
        f.write(imgdata)
        
    tmpimg = cv2.imread(filename)
    cv2.imwrite(filename, tmpimg)
        
    return filename;  


def save_annotation_file(folder, annotation_filename, annotaiton):
    os.makedirs(folder, exist_ok=True);
    f = open(os.path.join(folder, annotation_filename), 'w')
    f.write(annotaiton)
    f.close()
    
def saveAnnotation(folder, imageFileName, projectId, sourceImageSetId, annotations):
    # sourceImageSetId 폴더 생성
    os.makedirs(folder, exist_ok=True);
        
    annotationFileName = os.path.join(folder, str(sourceImageSetId) + '.csv');
    
    f = open(annotationFileName, 'a');
    
    for annotation in annotations:
        annotation_dict = json.loads(annotation['boundingBox']);
        xmin = annotation_dict['x'];
        ymin = annotation_dict['y'];
        xmax = annotation_dict['x'] + annotation_dict['width'];
        ymax = annotation_dict['y'] + annotation_dict['height'];
        annotationString = imageFileName + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + annotation['label'];
        # print(annotationString);
        f.write(annotationString + '\r\n');
        
    f.close();
    
def inf(model_name, filename, width, height, model_dir, infimage_dir):
    try:
        weights = os.path.join(model_dir, model_name, 'blood_cell/weights/best.pt')
        label = os.path.join(model_dir, model_name, 'label.txt')
        
        d, f = os.path.split(filename)
        f, e = os.path.splitext(f)

        inf_project = d;
        inf_name = f
        inf_path = os.path.join(inf_project, inf_name)
        inf_target_filename = f
        inf_target_fileext = e
        inf_target_folder = os.path.join(infimage_dir)
        result_file = os.path.join(inf_path, 'labels', inf_target_filename) + '.txt';

        source = filename;

        if (os.path.exists(inf_path )):
            shutil.rmtree(inf_path)

        detect.run(
            weights = weights
            , source = source
            , name = inf_name
            , project = inf_project
            , save_txt = True
            , save_conf = True
            , save_crop = False    
            , imgsz = (width, height)
            , max_det = 10000
            , agnostic_nms = true
        )

        # label 파일이 없다면 인퍼런스 시 디텍션 된 클래스가 없다는 뜻
        if os.path.exists(result_file):
            result_file_stream = open(result_file)
            lines = result_file_stream.readlines()
        else:
            lines = []
            
        label_flie_stream = open(label)
        labels = label_flie_stream.readlines()[0].split(',')

        titles = ['class', 'x1', 'y1', 'x2', 'y2', 'prob']
        tmpdict = {}
        result_json = {'result_code':'OK', 'result':[]}

        for line in lines:
            tmp = line.split(' ');
            for i in range(len(tmp)):
                if i == 0:
                    tmpdict[titles[i]] = labels[int(tmp[i])]
                else:
                    tmpdict[titles[i]] = tmp[i].replace('\n', '')

            result_json['result'].append(copy.deepcopy(tmpdict))
            
            
        # 인퍼런스 결과 삭제 
        if (os.path.exists(inf_path)):
            shutil.rmtree(inf_path)

        # 이미지 삭제
        os.remove(filename)

        return json.dumps(result_json)

    except Exception as e:
        err = {
            'result_code':'ERR',
            'error_trace':traceback.format_exc()
        }
        
        return json.dumps(err)    

####################################################################################################
#
# find_boudingbox
#
# PARAMETERS
#    fileName : 바운딩박스를 추출할 이미지 파일 절대 경로
#    minimum_size : 바운딩박스로 인정할 최소 크기
#
# RETURN
#    바운딩 박스 리스트
#    [
#        {"x1": 77, "y1": 218, "x2": 110, "y2": 256}, 
#        {"x1": 165, "y1": 210, "x2": 224, "y2": 256}
#    ]
#
####################################################################################################
def find_boudingbox(fileName, minimum_size):
    # 이미지 로드
    img = cv2.imread(fileName)

    # 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 경계선 검출
    edges = cv2.Canny(gray, 100, 200)

    # 윤곽선 찾기
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []

    # 모든 윤곽선 주변에 Bounding Box 그리기
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > minimum_size and h > minimum_size:
            bounding_boxes.append({'x1':x, 'y1':y, 'x2':x+w, 'y2':y+h})
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    # 결과 보여주기
    # plt.imshow(img)
    # plt.show()
    return bounding_boxes
