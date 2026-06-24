import os
import cv2
import base64
import shutil
import json
import sys
import traceback
import copy

# # Yolo V5 소스 경로
# WORKING_DIR = "/app/yolo"
# sys.path.append(WORKING_DIR)
# from model import detect

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
# # YOLO 클래스 임포트

import importlib
# ultralytics 모듈 경로 설정
ultralytics_module_path = '/app/yolo/ultralytics/ultralytics/__init__.py'
# ultralytics_module_path = '/home/smile/work/u2dlp-pbs/u2pbsai-inf/yolo/ultralytics/ultralytics/__init__.py'
# ultralytics_module_path = '/home/smile/ai/work/u2dlp-pbs/u2pbsai-inf/yolo/ultralytics/ultralytics/__init__.py'


spec = importlib.util.spec_from_file_location("ultralytics", ultralytics_module_path)
ultralytics = importlib.util.module_from_spec(spec)
sys.modules["ultralytics"] = ultralytics
spec.loader.exec_module(ultralytics)

from ultralytics import YOLO
    
    
def inf_v10(model_name, filename, width, height, model_dir, infimage_dir): 
    try:
        # 모델 가중치 및 라벨 파일 경로 설정
        weights = os.path.join(model_dir, model_name, 'blood_cell/weights/best.pt')
        label_path = os.path.join(model_dir, model_name, 'label.txt')  # 클래스 라벨 파일 경로

        # YOLO 모델 로드
        model = YOLO(weights)

        # 클래스 라벨 로드
        with open(label_path, 'r') as label_file:
            class_labels = label_file.readline().strip().split(',')

        # 입력 파일 경로 설정
        d, f = os.path.split(filename)
        f, e = os.path.splitext(f)

        # runs 디렉토리 내에 이미지 이름으로 폴더 생성
        result_dir = os.path.join('runs', f)
        os.makedirs(result_dir, exist_ok=True)

        # 텍스트 파일 저장 경로 설정
        result_txt_path = os.path.join(result_dir, f"{f}_result.txt")

        # 예측 수행
        results = model.predict(source=filename, save=False, imgsz=(width, height), device='cpu')  # 입력 크기를 유지

        # 원본 이미지 로드
        img = cv2.imread(filename)
        img_height, img_width, _ = img.shape

        # 특정 클래스 확신도 필터링을 위한 설정
        target_classes = [
            'RBC', 'Echinocyte', 'Elliptocyte', 'TearDropCell', 'TargetCell', 
            'Schistocyte', 'Nucleated', 'Chromicity', 'Reticulocyte', 'Baso-Stip.', 
            'Howell-Jolly', 'Spherocyte', 'Rouleaux-F.', 'Acanthocyte', 
            'SickleCell', 'Dimorphism', 'Stomatocyte'
        ]
        confidence_thresholds = {cls_name: 0.6 for cls_name in target_classes}  # 각 클래스에 대한 확신도 기준 설정
        print("ewrwerwer")
        print(confidence_thresholds)
        # JSON 형식으로 반환할 결과 저장용 변수
        result_json = {'result_code': 'OK', 'result': []}
        tmpdict = {}

        # 텍스트 파일에 예측 결과 저장
        with open(result_txt_path, 'w') as f_txt:
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 좌표는 (x1, y1, x2, y2) 형식
                confs = result.boxes.conf.cpu().numpy()  # 확신도
                classes = result.boxes.cls.cpu().numpy()  # 클래스 ID

                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    conf = float(conf)
                    cls = int(cls)
                    class_name = class_labels[cls]  # 클래스 이름을 가져옴

                    # 해당 클래스 이름이 필터링 대상에 포함되는지 확인 후 확신도 필터 적용
                    if class_name in confidence_thresholds and conf < confidence_thresholds[class_name]:
                        continue  # 확신도가 기준치 이하일 경우 스킵

                    # 텍스트 파일에 좌표 및 클래스 정보 저장
                    f_txt.write(f"{class_name} {x1} {y1} {x2} {y2} {conf:.6f}\n")

                    # JSON 형식으로 변환하여 저장
                    tmpdict['class'] = class_name
                    tmpdict['x1'] = x1
                    tmpdict['y1'] = y1
                    tmpdict['x2'] = x2
                    tmpdict['y2'] = y2
                    tmpdict['prob'] = conf
                    result_json['result'].append(tmpdict.copy())

                    # 바운딩 박스 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 클래스와 확신도 표시
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 이미지 저장 경로 설정
        result_img_path = os.path.join(result_dir, f"{f}_result.jpg")
        cv2.imwrite(result_img_path, img)

        print(f"Result image saved at {result_img_path}")
        print(f"Result text file saved at {result_txt_path}")

        # 결과 파일 삭제
        if os.path.exists(result_txt_path):
            os.remove(result_txt_path)
            print(f"Deleted: {result_txt_path}")
        if os.path.exists(result_img_path):
            os.remove(result_img_path)
            print(f"Deleted: {result_img_path}")

        # 폴더 삭제
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
            print(f"Deleted folder: {result_dir}")

        return json.dumps(result_json)

    except Exception as e:
        err = {
            'result_code': 'ERR',
            'error_trace': traceback.format_exc()
        }
        return json.dumps(err)
    
    
# def inf_v10(model_name, filename, width, height, model_dir, infimage_dir):
#     try:
#         # 모델 가중치 및 라벨 파일 경로 설정
#         weights = os.path.join(model_dir, model_name, 'blood_cell/weights/best.pt')
#         label_path = os.path.join(model_dir, model_name, 'label.txt')  # 클래스 라벨 파일 경로

#         # YOLO 모델 로드
#         model = YOLO(weights)

#         # 클래스 라벨 로드
#         with open(label_path, 'r') as label_file:
#             class_labels = label_file.readline().strip().split(',')

#         # 입력 파일 경로 설정
#         d, f = os.path.split(filename)
#         f, e = os.path.splitext(f)

#         # runs 디렉토리 내에 이미지 이름으로 폴더 생성
#         result_dir = os.path.join('runs', f)
#         os.makedirs(result_dir, exist_ok=True)

#         # 텍스트 파일 저장 경로 설정
#         result_txt_path = os.path.join(result_dir, f"{f}_result.txt")

#         # 예측 수행
#         results = model.predict(source=filename, save=False, imgsz=(width, height), device='cpu')  # 입력 크기를 유지

#         # 원본 이미지 로드
#         img = cv2.imread(filename)
#         img_height, img_width, _ = img.shape

#         # 확신도 필터 설정 (예: 0.5 이상만 허용)
#         confidence_threshold = 0.5

#         # 원본 이미지 로드
#         img = cv2.imread(filename)
#         img_height, img_width, _ = img.shape

#         # JSON 형식으로 반환할 결과 저장용 변수
#         result_json = {'result_code': 'OK', 'result': []}
#         titles = ['class', 'x1', 'y1', 'x2', 'y2', 'prob']
#         tmpdict = {}

#         # 텍스트 파일에 예측 결과 저장
#         with open(result_txt_path, 'w') as f_txt:
#             for result in results:
#                 boxes = result.boxes.xyxy.cpu().numpy()  # 좌표는 (x1, y1, x2, y2) 형식
#                 confs = result.boxes.conf.cpu().numpy()  # 확신도
#                 classes = result.boxes.cls.cpu().numpy()  # 클래스 ID

#                 for box, conf, cls in zip(boxes, confs, classes):
#                     if conf < confidence_threshold:
#                         continue  # 확신도가 기준치 이하일 경우 스킵

#                     x1, y1, x2, y2 = map(int, box)
#                     conf = float(conf)
#                     cls = int(cls)
#                     class_name = class_labels[cls]  # 클래스 이름을 가져옴

#                     # 텍스트 파일에 좌표 및 클래스 정보 저장
#                     f_txt.write(f"{class_name} {x1} {y1} {x2} {y2} {conf:.6f}\n")

#                     # JSON 형식으로 변환하여 저장
#                     tmpdict['class'] = class_name
#                     tmpdict['x1'] = x1
#                     tmpdict['y1'] = y1
#                     tmpdict['x2'] = x2
#                     tmpdict['y2'] = y2
#                     tmpdict['prob'] = conf
#                     result_json['result'].append(tmpdict.copy())

#                     # 바운딩 박스 그리기
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                     # 클래스와 확신도 표시
#                     label = f"{class_name}: {conf:.2f}"
#                     cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # 결과 이미지 저장 경로 설정
#         result_img_path = os.path.join(result_dir, f"{f}_result.jpg")
#         cv2.imwrite(result_img_path, img)

#         print(f"Result image saved at {result_img_path}")
#         print(f"Result text file saved at {result_txt_path}")

#         # 결과 파일 삭제
#         if os.path.exists(result_txt_path):
#             os.remove(result_txt_path)
#             print(f"Deleted: {result_txt_path}")
#         if os.path.exists(result_img_path):
#             os.remove(result_img_path)
#             print(f"Deleted: {result_img_path}")

#         # 폴더 삭제
#         if os.path.exists(result_dir):
#             shutil.rmtree(result_dir)
#             print(f"Deleted folder: {result_dir}")

#         return json.dumps(result_json)

#     except Exception as e:
#         err = {
#             'result_code': 'ERR',
#             'error_trace': traceback.format_exc()
#         }
#         return json.dumps(err)    



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

#         if (os.path.exists(inf_path)):
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
#             , agnostic_nms = True
#         )

#         # label 파일이 없다면 인퍼런스 시 디텍션 된 클래스가 없다는 뜻
#         if os.path.exists(result_file):
#             result_file_stream = open(result_file)
#             lines = result_file_stream.readlines()
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

# 	# 인퍼런스 결과 삭제 
#         if (os.path.exists(inf_path)):
#             shutil.rmtree(inf_path)
	
# 	# 이미지 삭제
#         os.remove(filename)

#         return json.dumps(result_json)

#     except Exception as e:
#         err = {
#             'result_code':'ERR',
#             'error_trace':traceback.format_exc()
#         }
        
#         return json.dumps(err)    



#####