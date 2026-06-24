import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import shutil

def FileSplit(filename):
    d, f = os.path.split(filename)
    f, e = os.path.splitext(f)
    return d, f, e.replace('.', '')

def SplitImage(image_filename, annotation_filename, annotations_new, image_crop_size) :
    imagefolder, imageId, imageExt = FileSplit(image_filename)
    
    image = cv2.imread(image_filename)
    
    image_annotations = pd.read_csv(annotation_filename)
    # image_annotations.rename(columns = {
    #     'IMAGES': 'image'
    #     , 'BOUNDINGBOX_LEFT': 'xmin'
    #     , 'BOUNDINGBOX_RIGHT': 'xmax'
    #     , 'BOUNDINGBOX_TOP': 'ymin'
    #     , 'BOUNDINGBOX_BOTTOM': 'ymax'
    #     , 'NAME': 'label'}, inplace=True);
    # image_annotations = image_annotations[['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', '', '', '']]
    # image_annotations['label'] = image_annotations['label'].str.replace('DB:', '')
    
    split_folder = os.path.join(imagefolder, 'split')

    width, height, depth = image.shape
    
    if os.path.exists(split_folder) :
        shutil.rmtree(split_folder)
    os.mkdir(split_folder)

    for x in range(0, width - image_crop_size, image_crop_size):
        for y in range(0, height - image_crop_size, image_crop_size):
            crop_xmin = x
            crop_ymin = y
            crop_xmax = x + image_crop_size
            crop_ymax = y + image_crop_size

            # 이미지 파일 크롭 및 저장
            croped = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            croped_filename = os.path.join(split_folder, f'{str(imageId)}_{str(crop_xmin)}_{str(crop_ymin)}.{imageExt}')
            print(croped_filename)
            cv2.imwrite(croped_filename, croped)

            # 어노테이션 저장
            annotations = SplitAnnotation(image_annotations, croped_filename, crop_xmin, crop_ymin, crop_xmax, crop_ymax)
            for annotation in annotations :
                annotations_new.loc[len(annotations_new)] = annotation

def SplitAnnotation(annotations, croped_filename, crop_xmin, crop_ymin, crop_xmax, crop_ymax) :
    condition = (
        (annotations.xmin <= crop_xmax) &
        (annotations.xmax >= crop_xmin) &
        (annotations.ymax >= crop_ymin) &
        (annotations.ymin < crop_ymax)
    )

    annotations_intersection = annotations.loc[condition]
    result = []

    rect1 = { 'xmin': crop_xmin,'ymin': crop_ymin, 'xmax': crop_xmax, 'ymax': crop_ymax }
    for annotation in annotations_intersection.values:
        rect2 = { 'xmin': annotation[1],'ymin': annotation[2], 'xmax': annotation[3], 'ymax': annotation[4] }
        tmp = GetIntersection(rect1, rect2)
        result.append({
            'image': croped_filename,
            'xmin': tmp['xmin'] - crop_xmin,
            'ymin': tmp['ymin'] - crop_ymin, 
            'xmax': tmp['xmax'] - crop_xmin, 
            'ymax': tmp['ymax'] - crop_ymin, 
            'label': annotation[5],
            'source': annotation[6],
            'orderId': annotation[7],
            'classType': annotation[8]
        })
    return result

def GetIntersection(rect1, rect2) :
    intersection = {}

    if not (rect1['xmax'] < rect2['xmin'] or rect1['xmin'] > rect2['xmax'] or rect1['ymin'] > rect2['ymax'] or rect1['ymax'] < rect2['ymin']) :
        if rect2['xmin'] < rect1['xmin'] :
            intersection['xmin'] = rect1['xmin']
        else :
            intersection['xmin'] = rect2['xmin']

        if rect2['ymin'] < rect1['ymin'] :
            intersection['ymin'] = rect1['ymin']
        else :
            intersection['ymin'] = rect2['ymin']

        if rect2['xmax'] > rect1['xmax'] :
            intersection['xmax'] = rect1['xmax']
        else :
            intersection['xmax'] = rect2['xmax']

        if rect2['ymax'] > rect1['ymax'] :
            intersection['ymax'] = rect1['ymax']
        else :
            intersection['ymax'] = rect2['ymax']

    return intersection