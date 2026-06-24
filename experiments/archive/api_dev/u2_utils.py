import os
import cv2
import base64
import shutil
import json
import sys
import traceback
import copy

# Yolo V5 소스 경로
WORKING_DIR = "/home/smile/work/pbs04"
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

###########################
### RBC 이미지 분할 추론

# 이미지 분할 함수
def divide_image(image_path):
    image = cv2.imread(image_path)
    # 이미지를 2배로 확대
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    height, width = image.shape[:2]
    # 분할된 이미지와 메타데이터를 저장할 리스트
    divided_images = []
    meta_data = []
    
    # 이미지를 500x500 크기로 분할
    step_size = 500  # 이미지가 겹치지 않도록 step_size를 변경
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            # 분할 영역 지정
            y_end = min(height, y + 500)
            x_end = min(width, x + 500)
            
            # 이미지 분할
            divided_img = image[y:y_end, x:x_end]
            
            # 분할된 이미지와 메타데이터 저장
            divided_images.append(divided_img)
            meta_data.append({
                'x_offset': x,
                'y_offset': y,
                'height': y_end - y,
                'width': x_end - x
            })
            
    return divided_images, meta_data

# 분할된 이미지를 임시 디렉터리에 저장하는 함수
def save_divided_images_to_temp_dir(divided_images, meta_data, temp_dir):
    saved_image_paths = []
    
    # 임시 디렉터리가 없으면 생성
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    for i, (image, meta) in enumerate(zip(divided_images, meta_data)):
        # 이미지 파일명 생성
        filename = f"temp_image_{i}_{meta['x_offset']}_{meta['y_offset']}.jpg"
        
        # 이미지 저장 경로
        save_path = os.path.join(temp_dir, filename)
        
        # 이미지 저장, 테스트용 추가
        is_saved = cv2.imwrite(save_path, image)
        if is_saved:
            print(f"Image has been saved at {save_path}")  # 이미지 저장 성공을 출력
        else:
            print(f"Failed to save image at {save_path}")  # 이미지 저장 실패를 출력
        
        # 저장된 이미지 경로 저장
        saved_image_paths.append(save_path)
        
    return saved_image_paths


def restore_coordinates(detected_lines, meta_data):
    restored_lines = []
    
    for line in detected_lines:
        class_id, x, y, w, h, conf = map(float, line.strip().split(" "))
        
        # 메타데이터에서 분할된 이미지의 원본 이미지에서의 위치와 크기 정보 가져오기
        x_offset = meta_data['x_offset']
        y_offset = meta_data['y_offset']
        
        # 원본 이미지의 좌표로 복원
        x_restored = (x + x_offset) / 2  # 배율 복원
        y_restored = (y + y_offset) / 2  # 배율 복원
        
        # 복원된 정보로 새로운 라인 생성
        new_line = f"{class_id} {x_restored} {y_restored} {w/2} {h/2} {conf}"  # 너비와 높이도 배율 복원
        restored_lines.append(new_line)
        
    return restored_lines
    
# def inf(model_name, filename, width, height, model_dir, infimage_dir):
#     try:
#         weights = os.path.join(model_dir, model_name, 'blood_cell/weights/best.pt')
#         label = os.path.join(model_dir, model_name, 'label.txt')
        
#         d, f = os.path.split(filename)
#         f, e = os.path.splitext(f)

#         inf_project = d;
#         inf_name = f
#         inf_path = os.path.join(inf_project, inf_name)
#         inf_target_filename = f
#         inf_target_fileext = e
#         inf_target_folder = os.path.join(infimage_dir)
#         result_file = os.path.join(inf_path, 'labels', inf_target_filename) + '.txt';

#         source = filename;
#         print(filename);
#         if (os.path.exists(inf_path )):
#             shutil.rmtree(inf_path)

#         # 이미지 크기 확인    
#         image = cv2.imread(filename)
#         image_height, image_width = image.shape[:2]
        
#         # 임시 디렉토리 경로
#         temp_dir = "temp_images"
        
            
#         # 너비와 높이가 모두 1000 이상인 경우에만 분할
#         if image_width >= 1000 and image_height >= 1000:  
#             # 이미지 분할
#             divided_images, meta_data = divide_image(filename)
#             saved_image_paths = save_divided_images_to_temp_dir(divided_images, meta_data, temp_dir)

#             for img_path in saved_image_paths:  # 각 이미지 경로에 대해
#                 # 분할된 이미지에 대한 detect.run 호출
#                 detect.run(
#                     weights=weights,
#                     source=img_path,  # 이미지 경로를 직접 전달
#                     name=inf_name,
#                     project=inf_project,
#                     save_txt=True,
#                     save_conf=True,
#                     save_crop=True,
#                     imgsz=(width, height),
#                     max_det=10000
#                 )
                
#             # 임시 디렉터리 삭제 (선택적)
#             # shutil.rmtree(saved_image_paths)


#         else:
#             # 일반 detect.run 호출
#             detect.run(
#                 weights=weights,
#                 source=filename,
#                 name=inf_name,
#                 project=inf_project,
#                 save_txt=True,
#                 save_conf=True,
#                 save_crop=True,
#                 imgsz=(width, height),
#                 max_det=10000
#             )

            
#         # label 파일이 없다면 인퍼런스 시 디텍션 된 클래스가 없다는 뜻
#         if os.path.exists(result_file):
#             with open(result_file, 'r') as result_file_stream:
#                 lines = result_file_stream.readlines()
#         else:
#             lines = []
            
#         label_flie_stream = open(label)
#         labels = label_flie_stream.readlines()[0].split(',')

#         titles = ['class', 'x1', 'y1', 'x2', 'y2', 'prob']
#         tmpdict = {}
#         result_json = {'result_code':'OK', 'result':[]}

#         for line in lines:
#             tmp = line.split(' ');
#             for i in range(len(tmp)):
#                 if i == 0:
#                     tmpdict[titles[i]] = labels[int(tmp[i])]
#                 else:
#                     tmpdict[titles[i]] = tmp[i].replace('\n', '')

#             result_json['result'].append(copy.deepcopy(tmpdict))

#         # 인퍼런스 결과 삭제 
#         if (os.path.exists(inf_path)):
#             shutil.rmtree(inf_path)

#         # 이미지 삭제
#         os.remove(filename)
        
#         return json.dumps(result_json)

#     except Exception as e:
#         err = {
#             'result_code':'ERR',
#             'error_trace':traceback.format_exc()
#         }
        
#         return json.dumps(err)   

def inf_nms(model_name, filename, width, height, model_dir, infimage_dir):
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
            , agnostic_nms = True
            , iou_thres=0.45
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

    ### 클래스 통합 nms를 위한 inf ver2(자체 nms)
# def inf(model_name, filename, width, height, model_dir, infimage_dir):
#     try:
#         weights = os.path.join(model_dir, model_name, 'blood_cell/weights/best.pt')
#         label = os.path.join(model_dir, model_name, 'label.txt')
        
#         d, f = os.path.split(filename)
#         f, e = os.path.splitext(f)

#         inf_project = d;
#         inf_name = f
#         inf_path = os.path.join(inf_project, inf_name)
#         inf_target_filename = f
#         inf_target_fileext = e
#         inf_target_folder = os.path.join(infimage_dir)
#         result_file = os.path.join(inf_path, 'labels', inf_target_filename) + '.txt';

#         source = filename;

#         if (os.path.exists(inf_path )):
#             shutil.rmtree(inf_path)

#         detect.run(
#             weights = weights
#             , source = source
#             , name = inf_name
#             , project = inf_project
#             , save_txt = True
#             , save_conf = True
#             , save_crop = False    
#             , imgsz = (width, height)
#             , max_det = 10000
#             , agnostic_nms = True,
#         )

#         # label 파일이 없다면 인퍼런스 시 디텍션 된 클래스가 없다는 뜻
#         if os.path.exists(result_file):
#             result_file_stream = open(result_file)
#             lines = result_file_stream.readlines()
#         else:
#             lines = []
            

#         # 탐지 결과 파싱
#         detections = [parse_detection_line(line, labels) for line in lines]
#         boxes = [d['bbox'] for d in detections]
#         scores = [d['prob'] for d in detections]

#         # NMS 적용
#         iou_threshold = 0.5  # IOU 임계값, 필요에 따라 조정
#         kept_indices = nms(boxes, scores, iou_threshold)

#         # 중복 제거된 결과 생성
#         final_detections = [detections[i] for i in kept_indices]

#         result_json = {'result_code': 'OK', 'result': final_detections}
            
            
#         # 인퍼런스 결과 삭제 
#         if (os.path.exists(inf_path)):
#             shutil.rmtree(inf_path)

#         # 이미지 삭제
#         os.remove(filename)

#         return json.dumps(result_json)

#     except Exception as e:
#         err = {
#             'result_code':'ERR',
#             'error_trace':traceback.format_exc()
#         }
        
#         return json.dumps(err)    

# def parse_detection_line(line, labels):
#     parts = line.split(' ')
#     class_id = int(parts[0])
#     x_center, y_center, width, height, confidence = map(float, parts[1:])
#     x1 = x_center - width / 2
#     y1 = y_center - height / 2
#     x2 = x_center + width / 2
#     y2 = y_center + height / 2
#     return {'class': labels[class_id], 'bbox': [x1, y1, x2, y2], 'prob': confidence}

# def nms(boxes, scores, iou_threshold):
#     """
#     Apply Non-Maximum Suppression (NMS) to the bounding boxes.
#     """
#     # 박스들을 스코어에 따라 정렬
#     sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

#     # NMS 적용
#     keep = []
#     while sorted_indices:
#         current = sorted_indices.pop(0)
#         keep.append(current)

#         sorted_indices = [i for i in sorted_indices if calculate_iou(boxes[current], boxes[i]) < iou_threshold]

#     return keep

# def calculate_iou(box1, box2):
#     """
#     Calculate the Intersection Over Union (IOU) of two bounding boxes.
#     """
#     # 각 박스의 (x1, y1, x2, y2) 좌표 추출
#     x1_min, y1_min, x1_max, y1_max = box1
#     x2_min, y2_min, x2_max, y2_max = box2

#     # 교차 영역의 (x, y) 좌표 계산
#     intersect_x_min = max(x1_min, x2_min)
#     intersect_y_min = max(y1_min, y2_min)
#     intersect_x_max = min(x1_max, x2_max)
#     intersect_y_max = min(y1_max, y2_max)

#     # 교차 영역의 넓이 계산
#     intersect_area = max(0, intersect_x_max - intersect_x_min) * max(0, intersect_y_max - intersect_y_min)

#     # 각 박스의 넓이 계산
#     box1_area = (x1_max - x1_min) * (y1_max - y1_min)
#     box2_area = (x2_max - x2_min) * (y2_max - y2_min)

#     # IOU 계산
#     union_area = box1_area + box2_area - intersect_area
#     iou = intersect_area / union_area

#     return iou

    
    
    
    
###### n순위 출력용
# def inf(model_name, filename, width, height, model_dir, infimage_dir, multi_class_cnt=None):
#     try:
#         weights = os.path.join(model_dir, model_name, 'blood_cell/weights/best.pt')
#         label = os.path.join(model_dir, model_name, 'label.txt')
        
#         d, f = os.path.split(filename)
#         f, e = os.path.splitext(f)

#         inf_project = d;
#         inf_name = f
#         inf_path = os.path.join(inf_project, inf_name)
#         inf_target_filename = f
#         inf_target_fileext = e
#         inf_target_folder = os.path.join(infimage_dir)
#         result_file = os.path.join(inf_path, 'labels', inf_target_filename) + '.txt';

#         source = filename;

#         if (os.path.exists(inf_path )):
#             shutil.rmtree(inf_path)

#         detect.run(
#             weights = weights
#             , source = source
#             , name = inf_name
#             , project = inf_project
#             , save_txt = True
#             , save_conf = True
#             , save_crop = True    
#             , imgsz = (width, height)
#             , max_det = 10000
#             , multi_class_cnt= multi_class_cnt
#         )

#         # label 파일이 없다면 인퍼런스 시 디텍 션 된 클래스가 없다는 뜻
#         if os.path.exists(result_file):
#             result_file_stream = open(result_file)
#             lines = result_file_stream.readlines()
#         else:
#             lines = []
            
#         label_flie_stream = open(label)
#         labels = label_flie_stream.readlines()[0].split(',')

#         class_title = []
#         if multi_class_cnt is not None:
#             class_title = [f"class{i}" for i in range(2, 2 + multi_class_cnt-1)] + [f"prob{i}" for i in range(2, 2 + multi_class_cnt-1)]
        
#         titles = ['class', 'x1', 'y1', 'x2', 'y2', 'prob'] + class_title
#         print(titles)
#         tmpdict = {}
#         result_json = {'result_code':'OK', 'result':[]}

#         for line in lines:
#             tmp = line.split(' ');
#             for i in range(len(tmp)):
#                 if i == 0:
#                     tmpdict[titles[i]] = labels[int(tmp[i])]
#                 elif i >= 6 and 6+multi_class_cnt-1 > i:
#                     tmpdict[titles[i]] = labels[int(tmp[i])]
#                 else:
#                     tmpdict[titles[i]] = tmp[i].replace('\n', '')

#             result_json['result'].append(copy.deepcopy(tmpdict))
#             print(result_json)
#         return json.dumps(result_json)

#     except Exception as e:
#         err = {
#             'result_code':'ERR',
#             'error_trace':traceback.format_exc()
#         }
        
#         return json.dumps(err)    
#######

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
