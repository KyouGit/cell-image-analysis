import pandas as pd
import numpy as np
import random
import os
import shutil

import glob
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import traceback

####################################################################################################
#
# GPU 점검
#
####################################################################################################
def check_gpu():
    import torch
    print(torch.__version__)
    print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
####################################################################################################
#
# annotation_tmp dataframe 의 내용을 파라미터에 따라 필터링하여 리턴
#
# source_imageset_ids : 이미지 파일이름 (image 컬럼) 에 source_imageset_ids 문자열이 포함된 경우
# labels : 라벨 (label 컬럼) 에 labels 의 각 항목이 포함된 경우
# only_perfect_labeled_image : True - 이미지의 어노테이션에 labels 에 지정된 라벨만 들어있는 경우
# miminum_annotation_count : 이미지당 어노테이션 수가 miminum_annotation_count 개 이상인 경우
#
####################################################################################################
def pandas_filter(annotation_tmp, source_imageset_ids, labels, only_perfect_labeled_image, miminum_annotation_count):
    condition = (
        ((len(source_imageset_ids) == 0) | annotation_tmp['image'].str.contains('|'.join(source_imageset_ids))) &
        ((len(labels) == 0) | annotation_tmp['label'].isin(labels))
    )

    if only_perfect_labeled_image and len(labels) > 0:
        # 라벨 필터가 존재 할 때 해당 이미지에 모든 라벨이 존재하는 경우에만 선택

        # 라벨이 이미지당 n개 이상인 이미지 목록
        annotation_tmp_label_cnt = annotation_tmp.groupby(['image']).size().reset_index(name='count')
        annotation_tmp_label_cnt = annotation_tmp_label_cnt[annotation_tmp_label_cnt['count'] >= miminum_annotation_count]

        # 지정된 라벨만 들어있는 이미지 목록
        annotation_tmp_label_distinct_cnt = pd.merge(
            annotation_tmp.groupby(['image'])['label'].nunique(),
            annotation_tmp[condition].groupby(['image'])['label'].nunique(),
            on='image', how='inner'
        )
        annotation_tmp_label_distinct_cnt = annotation_tmp_label_distinct_cnt[annotation_tmp_label_distinct_cnt['label_x'] == annotation_tmp_label_distinct_cnt['label_y']]

        # 타겟 이미지 목록 (이미지당 n개 이상 및 지정된 라벨만 들어있는 이미지 목록)
        annotation_image_target = pd.merge(
            annotation_tmp_label_cnt,
            annotation_tmp_label_distinct_cnt,
            on='image', how='inner'
        )

        annotation_tmp_filtered = pd.merge(
            annotation_tmp[condition],
            annotation_image_target,
            on='image', how='inner'
        )
        
        return annotation_tmp_filtered
    else:
        return annotation_tmp[condition]
    
####################################################################################################
#
# 학습을 위한 어노테이션을 추출하여 pandas dataframe 으로 리턴
#
# PARAMETERS #
#
# source_image_dir : 어노테이션 csv 파일이 존재하는 폴더
#     source_image_dir 폴더 내의 모든 csv 파일을 찾은 뒤 다음 조건에 따라 필터링하여 
#     하나의 dataframe 으로 리턴 함
# source_imageset_ids : 이미지 파일이름 (image 컬럼) 에 source_imageset_ids 문자열이 포함된 경우
# labels : 라벨 (label 컬럼) 에 labels 의 각 항목이 포함된 경우
# only_perfect_labeled_image : True - 이미지의 어노테이션에 labels 에 지정된 라벨만 들어있는 경우
# miminum_annotation_count : 이미지당 어노테이션 수가 miminum_annotation_count 개 이상인 경우
# csv_filename : 어노테이션 csv 파일 이름을 직접 지정 함
#     csv_filename 이 지정 된 경우 source_image_dir 내 전체 csv 파일을 찾지 않고
#     csv_filename 파일 하나만 읽은 뒤 필터링 조건을 적용 하여 리턴 함
#     DI60 의 WBC, RBC 인 경우에 사용 (매번 csv 파일을 머지하는 시간을 없애기 위함)
#
# EXAMPLE #
#
# TRAIN_ID = '230320_DI_001_2'
# SOURCE_IMAGE_DIR = '/home/smile/work/pbs/dataset/di60' # DI60 검사모드용
# source_imageset_ids = ['test_2303092710942_RBC', 'test_2303092807812_RBC', 'test_2303101703162_RBC', 'test_2303112709922_RBC', 'test_2303131506462_RBC', 'test_2303131701502_RBC', 'test_2303131701512_RBC', 'test_2303132100522_RBC', 'test_2303132708612_RBC', 'test_2303142202932_RBC', 'test_2303142710922_RBC']
# labels = ['Acanthocytosis', 'Echinocytosis' , 'Normal shape', 'Ovalocytosis', 'Teardrop cells']
# only_perfect_labeled_image = True
# miminum_annotation_count = 0
# csv_filename = '/home/smile/work/pbs/dataset/di60/di60_RBC/merged.csv'
# annotation = select_source(SOURCE_IMAGE_DIR, source_imageset_ids, labels, only_perfect_labeled_image, miminum_annotation_count, csv_filename)
#
####################################################################################################
def select_source(source_image_dir, source_imageset_ids, labels, only_perfect_labeled_image, miminum_annotation_count, csv_filename = '', exceptValSet = False) :
    csvlist = [];
    
    if csv_filename != '':
        annotation_tmp = pd.read_csv(csv_filename)
        annotation_all = pandas_filter(annotation_tmp, source_imageset_ids, labels, only_perfect_labeled_image, miminum_annotation_count)
    else:
        # 합쳐진 어노테이션 테이블 만들기
        for path, dirs, files in os.walk(source_image_dir):
            if exceptValSet == False or (path != '/home/smile/work/pbs/dataset/u2labeler/48/88' and path != '/home/smile/work/pbs/dataset/u2labeler/48/112'):            
                for file in files:
                    name, ext = os.path.splitext(file);
                    if 'checkpoint' not in file and ext == '.csv':
                        csvlist.append(os.path.join(path, file));

        annotation_all = pd.DataFrame()
        for csv in tqdm(csvlist, desc='merge csv'):
            annotation_tmp = pd.read_csv(csv);

            try:
                annotation_filtered = pandas_filter(annotation_tmp, source_imageset_ids, labels, only_perfect_labeled_image, miminum_annotation_count)
                annotation_all = pd.concat([annotation_all, annotation_filtered])
            except Exception:
                print('error')
                print(csv)
                print(traceback.format_exc())

                
    print('전체 라벨 수')
    print(annotation_all.groupby('label').count()['image'])
        
    # 추출 조건
    condition = (
        ((len(source_imageset_ids) == 0) | annotation_all['image'].str.contains('|'.join(source_imageset_ids))) &
        ((len(labels) == 0) | annotation_all['label'].isin(labels))
    )   

    annotation_all = annotation_all[condition].reindex(columns=['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
    
    return annotation_all
    
####################################################################################################
#
# 폴더 내 csv 파일을 합친 파일 생성하기
# di60 데이터를 업로드 한 뒤 csv 파일을 머지하기 위하여 한 번 호출 
# wbc 및 rbc 두 개의 파일 사용
#
#    WBC : /home/smile/work/pbs/dataset/di60/merged.csv
#    RBC : /home/smile/work/pbs/dataset/di60_RBC/merged.csv
#
####################################################################################################
def merge_csv_to_file(source_image_dir, merged_csv_filename):
    annotation_all = merge_csv(source_image_dir)        
    annotation_all.to_csv(merged_csv_filename, index=False)
    
def merge_csv(source_image_dir):
    csvlist = []
    annotation_all = pd.DataFrame()
    
    for path, dirs, files in os.walk(source_image_dir):
        for file in files:
            name, ext = os.path.splitext(file);
            if 'checkpoint' not in file and ext == '.csv':
                csvlist.append(os.path.join(path, file));
                
    for csv in tqdm(csvlist):
        annotation = pd.read_csv(csv)
        annotation_all = pd.concat([annotation_all, annotation]);
        
    return annotation_all
    
####################################################################################################
#
# 학습 및 검증 데이터 분할 목록 생성
#
# PARAMETERS #
#
# annotation : 어노테이션 dataframe
# train_rate : 학습 데이터 비중 (0 ~ 1)
# labels : 소스 라벨 필터
#
####################################################################################################    
def split_test_train(annotation, train_rate, labels, maximum_image_count):
    # 학습용/검증용 이미지 수
    images = annotation.image.unique()
    
    if maximum_image_count > 0:
        images = images[:maximum_image_count]
    
    train_count = math.trunc(len(images) * train_rate)
    
    # 학습용/검증용 나누기 (랜덤)
    train_images_fullpath = images[random.sample(list(range(len(images))), train_count)]
    test_images_fullpath = list(set(images) - set(train_images_fullpath))
    
    train_images = list(map(lambda filename: os.path.split(filename)[1], train_images_fullpath))
    test_images = list(map(lambda filename: os.path.split(filename)[1], test_images_fullpath))

    # 학습용/검증용 이미지 및 라벨 정보 출력  
    print(f'전체 이미지 수 : {len(images)}')
    print(f'트레이닝 이미지 수 : {train_count}')
    print(f'검증 이미지 수 : {len(images) - train_count}')
    print('선택된 라벨 수')
    print(annotation.groupby('label').count()[['image']])
    
    # 학습용/검증용 인스턴스 수
    condition = (
        (annotation['image'].str.contains('|'.join(train_images_fullpath))) &
        ((len(labels) == 0) | annotation['label'].isin(labels))
    )

    train_annotation = annotation[condition]

    condition = (
        (annotation['image'].str.contains('|'.join(test_images_fullpath))) &
        ((len(labels) == 0) | annotation['label'].isin(labels))
    )

    test_annotation = annotation[condition]

    print('\r\n학습용 라벨 수')
    print(train_annotation.groupby('label').count()[['image']])
    print('\r\n검증용 라벨 수')
    print(test_annotation.groupby('label').count()[['image']])
    
    return train_images, train_images_fullpath, test_images, test_images_fullpath

####################################################################################################
#
# 어노테이션 dataframe 을 이용하여 학습용 및 테스트용 데이터를 나누어 준비 
#     1. 이미지 크기 확인
#     2. 학습 및 검증 데이터 분할 목록 생성
#     3. 라벨 목록 생성
#     4. 학습용 및 테스트용 이미지와 라벨 생성
#     5. 학습용 및 테스트용 이미지와 라벨을 샘플링하여 화면에 출력
#
# PARAMETERS #
#
# annotation : 어노테이션 dataframe
# train_rate : 학습 데이터 비중 (0 ~ 1)
# labels : 소스 라벨 필터
# train_label_dir : 학습용 라벨이 생성될 폴더
# train_image_dir : 학습용 이미지가 생성될 폴더
# test_label_dir : 검증용 라벨이 생성될 폴더
# test_image_dir : 검증용 이미지가 생성될 폴더
# imagefile_ext : 이미지 파일 확장자
#     
####################################################################################################
def prepare_training_data(annotation, train_rate, labels, train_label_dir, train_image_dir, test_label_dir, test_image_dir, imagefile_ext, show_sample_count=4, maximum_image_count=0):
    ####################################################################################################
    #
    # 이미지 크기 확인
    #
    ####################################################################################################
    width, height = check_image_size(annotation['image'].values[0])

    ####################################################################################################
    #
    # 학습 및 검증 데이터 분할 목록 생성
    #
    ####################################################################################################
    train_images, train_images_fullpath, test_images, test_images_fullpath = split_test_train(annotation, train_rate, labels, maximum_image_count)
    
    ####################################################################################################
    #
    # 라벨 목록 생성
    #
    ####################################################################################################
    cells_id, cells_classes = get_label_dictionary(labels)

    ####################################################################################################
    #
    # 확인용 코드
    #
    ####################################################################################################
    # annotation 에 지정된 이미지가 잘 로딩 되는지 확인
    # image = cv2.imread(annotation['image'].values[0])
    # image = image[:,:,2::-1]
    # plt.imshow(image)

    ####################################################################################################
    #
    # 학습용 및 테스트용 이미지와 라벨 생성
    #
    ####################################################################################################
    # 학습용 데이터 (이미지 및 라벨) 생성
    create_train_data('train', train_images_fullpath, annotation, cells_id, train_label_dir, train_image_dir, width, height)
    # 테스트용 데이터 (이미지 및 라벨) 생성
    create_train_data('test', test_images_fullpath, annotation, cells_id, test_label_dir, test_image_dir, width, height)                      

    ####################################################################################################
    #
    # 학습용 및 테스트용 이미지와 라벨을 샘플링하여 확인
    #     학습용 4개 (상단)
    #     테스트용 4개 (하단)
    #
    ####################################################################################################
    sample_train_image_files = random.sample(train_images, show_sample_count)
    merged_train_image = merge_images_with_label(sample_train_image_files, imagefile_ext, cells_classes, train_image_dir, train_label_dir, width, height)
    sample_test_image_files = random.sample(test_images, show_sample_count)
    merged_test_image = merge_images_with_label(sample_test_image_files, imagefile_ext, cells_classes, test_image_dir, test_label_dir, width, height)
    merged_image = np.concatenate((merged_train_image, merged_test_image), axis=0)
    merged_image = merged_image[:,:,2::-1]

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(merged_image); 
    
    return width, height, cells_classes, train_images, test_images

####################################################################################################
#
# 학습용 라벨 목록 생성
#
# PARAMETERS #
#
# annotation : 어노테이션 dataframe
#
####################################################################################################   
#def get_label_dictionary(annotation):
def get_label_dictionary(labels):
    # 라벨 딕셔너리 생성
    cells_id = {}
    # for i, label in enumerate(annotation.label.unique()):
    #     cells_id[label] = i;
    
    for i, label in enumerate(labels):
        cells_id[label] = i;

    cells_classes = list(cells_id.keys())
    print(f'라벨 : {cells_classes}')
    
    return cells_id, cells_classes

def replace_image(src_file, dst_file, size):
    image = cv2.imread(src_file) 
    image = cv2.resize(image, size)
    cv2.imwrite(dst_file, image)
    
# 이미지 뷰
def draw_image(image_file, label_file, class_names, width, height):   
    image = cv2.imread(image_file)
        
    with open(label_file) as fobj:
        while True:            
            item = fobj.readline()
            if item is None or len(item)<=0:
                break
                
            item = item.split()
            
            lb = int(item[0])
            xc = float(item[1]) * width
            yc = float(item[2]) * height
            w = float(item[3]) * width
            h = float(item[4]) * height
            
            # print(xc, yc, w, h)
        
            image = cv2.rectangle(image, (int(xc - w/2), int(yc - h/2)), (int(xc + w/2), int(yc + h/2)), (0,0,255), 1)
            image = cv2.putText(image, class_names[lb], (int(xc - w/2), int(yc - h/2 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (40, 40, 40), 1)
        
    return image

def merge_images_with_label(image_files, image_file_ext, cells_classes, image_dir, label_dir, width, height):
    merged_image = None
    
    for sample_image_file in image_files:
        image_file = os.path.join(image_dir, sample_image_file)
        label_file = os.path.join(label_dir, sample_image_file.replace(image_file_ext, ".txt"))

        image = draw_image(image_file, label_file, cells_classes, width, height)

        if merged_image is None:
            merged_image = image
        else:
            merged_image = np.concatenate((merged_image, image), axis=1)

    
    
    return merged_image

def check_image_size(sample_image_file):
    image = cv2.imread(sample_image_file)
    image = image[:,:,2::-1]
    width, height, depth = image.shape
    print(f'샘플이미지: {sample_image_file}')
    print(f'이미지 크기: {width} X {height}')
    
    return width, height

####################################################################################################
#
# 선택된 이미지 리스트 (image_fullpath) 를 이용하여 학습 혹은 테스트용 이미지 및 라벨을 생성 함
#
# PARAMETERS #
#
# job : train 또는 test. tqdm title 용
# images_fullpath : 학습 혹은 테스트용 이미지 파일 이름 목록
# annotation : 어노테이션 dataframe
# cells_id : 라벨 목록
# label_dir : 라벨데이터를 생성할 폴더
# image_dir : 이미지를 생성할 폴더
# width : 이미지 너비
# height : 이미지 높이
#
####################################################################################################
def create_train_data(job, images_fullpath, annotation, cells_id, label_dir, image_dir, width, height):
    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
    os.makedirs(label_dir, exist_ok=True)
    
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir, exist_ok=True)
    
    for image in tqdm(images_fullpath, desc='create ' + job + ' label'):
        path, name = os.path.split(image);
        name, ext = os.path.splitext(name);
        lables_file = os.path.join(label_dir, name + ".txt")    
                      
        with open(lables_file, "w") as wobj:
            for box in annotation.loc[annotation.image == image].values:
                wobj.write("%d %f %f %f %f \n" % (
                    cells_id[box[5]],
                    ((box[3]+box[1])/2.0) / width,
                    ((box[4]+box[2])/2.0) / height,
                    (box[3]-box[1]) / width,
                    (box[4]-box[2]) / height
                    # 0.5, 0.5, 1, 1
                ))                      
                      
    size = (width, height)

    for image in tqdm(images_fullpath, desc='create ' + job + ' image'):
        path, name = os.path.split(image);
        src_file = image
        dst_file = os.path.join(image_dir, name)
        replace_image(src_file, dst_file, size)