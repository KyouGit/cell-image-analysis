"""
High-Resolution Image Splitter with Annotation Mapping

Splits large microscopy images into fixed-size tiles (e.g., 224x224)
while correctly mapping bounding box annotations to the split coordinates.

Used in the U2Bio pipeline to prepare training data from high-resolution
whole-slide blood smear images.
"""

import pandas as pd
import cv2
import os
import shutil


def FileSplit(filename):
    """Split filepath into directory, name, extension"""
    d, f = os.path.split(filename)
    f, e = os.path.splitext(f)
    return d, f, e.replace('.', '')


def SplitImage(image_filename, annotation_filename, annotations_new, image_crop_size):
    """
    Split a large image into tiles and remap annotations.

    Args:
        image_filename: Path to source image
        annotation_filename: Path to annotation CSV (image,xmin,ymin,xmax,ymax,label)
        annotations_new: DataFrame to append split annotations to
        image_crop_size: Tile size (e.g., 224)
    """
    imagefolder, imageId, imageExt = FileSplit(image_filename)

    image = cv2.imread(image_filename)
    image_annotations = pd.read_csv(annotation_filename)

    split_folder = os.path.join(imagefolder, 'split')
    width, height, depth = image.shape

    if os.path.exists(split_folder):
        shutil.rmtree(split_folder)
    os.mkdir(split_folder)

    for x in range(0, width - image_crop_size, image_crop_size):
        for y in range(0, height - image_crop_size, image_crop_size):
            crop_xmin, crop_ymin = x, y
            crop_xmax = x + image_crop_size
            crop_ymax = y + image_crop_size

            # Crop and save tile
            cropped = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            cropped_filename = os.path.join(split_folder,
                                            f'{imageId}_{crop_xmin}_{crop_ymin}.{imageExt}')
            cv2.imwrite(cropped_filename, cropped)

            # Remap annotations to tile coordinates
            annotations = SplitAnnotation(image_annotations, cropped_filename,
                                          crop_xmin, crop_ymin, crop_xmax, crop_ymax)
            for annotation in annotations:
                annotations_new.loc[len(annotations_new)] = annotation


def SplitAnnotation(annotations, cropped_filename, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
    """Find and remap annotations that intersect with the crop region"""
    condition = (
        (annotations.xmin <= crop_xmax) &
        (annotations.xmax >= crop_xmin) &
        (annotations.ymax >= crop_ymin) &
        (annotations.ymin < crop_ymax)
    )
    annotations_intersection = annotations.loc[condition]
    result = []

    rect1 = {'xmin': crop_xmin, 'ymin': crop_ymin, 'xmax': crop_xmax, 'ymax': crop_ymax}
    for annotation in annotations_intersection.values:
        rect2 = {'xmin': annotation[1], 'ymin': annotation[2],
                  'xmax': annotation[3], 'ymax': annotation[4]}
        tmp = GetIntersection(rect1, rect2)
        result.append({
            'image': cropped_filename,
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


def GetIntersection(rect1, rect2):
    """Compute intersection of two rectangles"""
    if (rect1['xmax'] < rect2['xmin'] or rect1['xmin'] > rect2['xmax'] or
            rect1['ymin'] > rect2['ymax'] or rect1['ymax'] < rect2['ymin']):
        return {}

    return {
        'xmin': max(rect1['xmin'], rect2['xmin']),
        'ymin': max(rect1['ymin'], rect2['ymin']),
        'xmax': min(rect1['xmax'], rect2['xmax']),
        'ymax': min(rect1['ymax'], rect2['ymax'])
    }
