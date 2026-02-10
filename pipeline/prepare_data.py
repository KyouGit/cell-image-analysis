"""
BCCD Dataset Preparation: Download + VOC→YOLO Format Conversion

Downloads the BCCD (Blood Cell Count and Detection) dataset and converts
Pascal VOC XML annotations to YOLO detection format.

Usage:
    python prepare_data.py

Output:
    data/bccd_yolo/
        images/train/    (~229 images)
        images/val/      (~60 images)
        images/test/     (~75 images)
        labels/train/    (YOLO format .txt)
        labels/val/
        labels/test/
        data.yaml
"""

import os
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# Class mapping: BCCD dataset has 3 cell types
CLASS_MAP = {
    'WBC': 0,
    'RBC': 1,
    'Platelets': 2,  # BCCD XML uses "Platelets"
}

# Normalized class names for output directories
CLASS_NAMES = ['WBC', 'RBC', 'Platelet']


def clone_bccd(dest_dir='data/BCCD_Dataset'):
    """Clone BCCD dataset from GitHub"""
    dest = Path(dest_dir)
    if dest.exists():
        print(f"[INFO] BCCD dataset already exists at {dest}")
        return dest

    print("[INFO] Cloning BCCD dataset...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'clone', 'https://github.com/Shenggan/BCCD_Dataset.git', str(dest)],
        check=True
    )
    print(f"[INFO] Cloned to {dest}")
    return dest


def parse_voc_xml(xml_path):
    """
    Parse Pascal VOC XML annotation file

    Returns:
        img_size: (width, height)
        objects: list of (class_name, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append((name, xmin, ymin, xmax, ymax))

    return (width, height), objects


def voc_to_yolo(img_size, bbox):
    """
    Convert VOC bbox (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, w, h)
    All values normalized to [0, 1]
    """
    w_img, h_img = img_size
    xmin, ymin, xmax, ymax = bbox

    x_center = (xmin + xmax) / 2.0 / w_img
    y_center = (ymin + ymax) / 2.0 / h_img
    w = (xmax - xmin) / float(w_img)
    h = (ymax - ymin) / float(h_img)

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return x_center, y_center, w, h


def read_split_file(split_path):
    """Read image IDs from BCCD split file (train.txt, val.txt, test.txt)"""
    ids = []
    with open(split_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract just the filename without path/extension
                name = Path(line).stem
                ids.append(name)
    return ids


def convert_dataset(bccd_dir='data/BCCD_Dataset', output_dir='data/bccd_yolo'):
    """
    Convert BCCD VOC dataset to YOLO detection format

    Uses existing train/val/test splits from the BCCD dataset.
    """
    bccd_dir = Path(bccd_dir)
    output_dir = Path(output_dir)

    annotations_dir = bccd_dir / 'BCCD' / 'Annotations'
    images_dir = bccd_dir / 'BCCD' / 'JPEGImages'
    splits_dir = bccd_dir / 'BCCD' / 'ImageSets' / 'Main'

    # Verify paths exist
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found: {images_dir}")

    # Read splits
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f'{split_name}.txt'
        if split_file.exists():
            splits[split_name] = read_split_file(split_file)
            print(f"[INFO] {split_name}: {len(splits[split_name])} images")
        else:
            print(f"[WARNING] Split file not found: {split_file}")

    if not splits:
        raise FileNotFoundError("No split files found")

    # Create output structure
    stats = {split: {'images': 0, 'objects': {name: 0 for name in CLASS_NAMES}} for split in splits}
    skipped_classes = set()

    for split_name, image_ids in splits.items():
        img_out_dir = output_dir / 'images' / split_name
        lbl_out_dir = output_dir / 'labels' / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_id in image_ids:
            xml_path = annotations_dir / f'{img_id}.xml'
            if not xml_path.exists():
                print(f"[WARNING] Annotation not found: {xml_path}")
                continue

            # Find the image file (try common extensions)
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = images_dir / f'{img_id}{ext}'
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"[WARNING] Image not found for: {img_id}")
                continue

            # Parse XML
            img_size, objects = parse_voc_xml(xml_path)

            # Convert to YOLO format
            yolo_lines = []
            for class_name, xmin, ymin, xmax, ymax in objects:
                if class_name not in CLASS_MAP:
                    skipped_classes.add(class_name)
                    continue

                class_id = CLASS_MAP[class_name]
                x_c, y_c, w, h = voc_to_yolo(img_size, (xmin, ymin, xmax, ymax))
                yolo_lines.append(f'{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}')

                # Update stats
                stats[split_name]['objects'][CLASS_NAMES[class_id]] += 1

            # Write YOLO label file
            label_path = lbl_out_dir / f'{img_id}.txt'
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            # Copy image
            dst_img = img_out_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            stats[split_name]['images'] += 1

    if skipped_classes:
        print(f"[WARNING] Skipped unknown classes: {skipped_classes}")

    return output_dir, stats


def create_data_yaml(output_dir='data/bccd_yolo'):
    """Create YOLO data.yaml configuration file"""
    output_dir = Path(output_dir)
    yaml_path = output_dir / 'data.yaml'

    # Use relative path to avoid Korean character encoding issues
    yaml_content = f"""# BCCD Blood Cell Detection Dataset
# 3 classes: WBC, RBC, Platelet

path: {output_dir}
train: images/train
val: images/val
test: images/test

nc: 3
names: ['WBC', 'RBC', 'Platelet']
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"[INFO] Created {yaml_path}")
    return yaml_path


def print_stats(stats):
    """Print conversion statistics"""
    print("\n" + "=" * 60)
    print("Dataset Conversion Summary")
    print("=" * 60)

    total_images = 0
    total_objects = {name: 0 for name in CLASS_NAMES}

    for split_name, split_stats in stats.items():
        print(f"\n  {split_name}:")
        print(f"    Images: {split_stats['images']}")
        total_images += split_stats['images']
        for cls_name, count in split_stats['objects'].items():
            print(f"    {cls_name}: {count} objects")
            total_objects[cls_name] += count

    print(f"\n  Total:")
    print(f"    Images: {total_images}")
    for cls_name, count in total_objects.items():
        print(f"    {cls_name}: {count} objects")
    print("=" * 60)


def main():
    print("=" * 60)
    print("BCCD Dataset Preparation")
    print("=" * 60)

    # 1. Clone dataset
    print("\n[Step 1] Downloading BCCD dataset...")
    bccd_dir = clone_bccd('data/BCCD_Dataset')

    # 2. Convert VOC → YOLO
    print("\n[Step 2] Converting VOC XML to YOLO format...")
    output_dir, stats = convert_dataset(str(bccd_dir), 'data/bccd_yolo')

    # 3. Create data.yaml
    print("\n[Step 3] Creating data.yaml...")
    yaml_path = create_data_yaml(str(output_dir))

    # 4. Print summary
    print_stats(stats)

    print(f"\nDataset ready at: {output_dir}")
    print(f"YOLO config: {yaml_path}")
    print("\nNext step: python yolo_detect.py --mode train")


if __name__ == '__main__':
    main()
