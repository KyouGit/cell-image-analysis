"""
Detection-based Demo: YOLO Detection + Similarity Search

Usage:
    # Detect and visualize bounding boxes on an image
    python example.py detect path/to/image.jpg

    # Find similar cropped cells
    python example.py similar path/to/cell_crop.jpg

    # Quick latent space analysis on cropped cells
    python example.py analyze
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def detect_and_visualize(image_path, model_path='runs/detect/bccd/weights/best.pt',
                         conf_threshold=0.25, save_path='outputs/detection_result.png'):
    """
    Run YOLO detection on an image and visualize bounding boxes

    Args:
        image_path: Path to input image
        model_path: Path to trained YOLO model
        conf_threshold: Minimum confidence threshold
        save_path: Output path for visualization
    """
    from ultralytics import YOLO

    CLASS_NAMES = ['WBC', 'RBC', 'Platelet']
    COLORS = {'WBC': '#2196F3', 'RBC': '#F44336', 'Platelet': '#4CAF50'}

    print(f"[INFO] Detecting cells in: {image_path}")

    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_threshold, verbose=False)

    # Load image for matplotlib
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img)

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        counts = {name: 0 for name in CLASS_NAMES}

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

            if cls_id >= len(CLASS_NAMES):
                continue

            cls_name = CLASS_NAMES[cls_id]
            color = COLORS[cls_name]
            counts[cls_name] += 1

            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'{cls_name} {conf:.2f}',
                    fontsize=8, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

        # Print detection summary
        total = sum(counts.values())
        print(f"[RESULTS] {total} cells detected:")
        for name, count in counts.items():
            print(f"  {name}: {count}")
    else:
        print("[INFO] No detections found")

    ax.set_title('YOLO Blood Cell Detection', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved: {save_path}")
    plt.close()


def find_similar_cells(query_path, data_dir='data/cropped_cells',
                       model_path='outputs/best_autoencoder.pth',
                       top_k=5, save_path='outputs/similar_cells.png'):
    """
    Find similar cell images using AutoEncoder latent space

    Args:
        query_path: Path to query cell crop image
        data_dir: Directory with cropped cells
        model_path: Path to trained AutoEncoder
        top_k: Number of similar images to find
        save_path: Output visualization path
    """
    from main import ConvAutoEncoder, CellImageDataset, extract_features, get_val_transform
    from torch.utils.data import DataLoader
    from scipy.spatial.distance import cdist

    print(f"[INFO] Finding cells similar to: {query_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoEncoder(latent_dim=256)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Use correct [-1, 1] normalization (matching training)
    transform = get_val_transform(128)

    # Encode query image
    query_img = Image.open(query_path).convert('RGB')
    query_tensor = transform(query_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, query_features = model(query_tensor)
        query_features = query_features.cpu().numpy()

    # Extract all features
    dataset = CellImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    all_features, all_labels, all_paths = extract_features(model, dataloader, device)

    # Compute distances
    distances = cdist(query_features, all_features, metric='euclidean')[0]
    top_indices = np.argsort(distances)[:top_k]

    # Print results
    print(f"\nTop {top_k} similar cells:")
    print("-" * 60)
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {all_paths[idx]}")
        print(f"     Distance: {distances[idx]:.4f}, Class: {dataset.idx_to_class[all_labels[idx]]}")

    # Visualize
    fig, axes = plt.subplots(1, top_k + 1, figsize=(top_k * 2.5 + 3, 3))

    axes[0].imshow(query_img)
    axes[0].set_title('Query', fontweight='bold', fontsize=10)
    axes[0].axis('off')

    for i, idx in enumerate(top_indices):
        img = Image.open(all_paths[idx])
        axes[i + 1].imshow(img)
        cls_name = dataset.idx_to_class[all_labels[idx]]
        axes[i + 1].set_title(f'#{i+1} {cls_name}\nDist: {distances[idx]:.2f}', fontsize=9)
        axes[i + 1].axis('off')

    plt.suptitle('Similar Cell Search (AutoEncoder Latent Space)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Saved: {save_path}")
    plt.close()


def quick_analysis(data_dir='data/cropped_cells', model_path='outputs/best_autoencoder.pth',
                   max_per_class=500):
    """Quick latent space visualization using UMAP"""
    from main import ConvAutoEncoder, CellImageDataset, extract_features, get_val_transform
    from torch.utils.data import DataLoader
    import umap

    print("=" * 60)
    print("Quick Latent Space Analysis")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_val_transform(128)

    dataset = CellImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = ConvAutoEncoder(latent_dim=256)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    features, labels, paths = extract_features(model, dataloader, device)

    # Subsample for balanced visualization
    if max_per_class is not None:
        selected_idx = []
        rng = np.random.RandomState(42)
        for label in np.unique(labels):
            class_idx = np.where(labels == label)[0]
            if len(class_idx) > max_per_class:
                class_idx = rng.choice(class_idx, max_per_class, replace=False)
            selected_idx.extend(class_idx)
        selected_idx = np.array(sorted(selected_idx))
        features = features[selected_idx]
        labels = labels[selected_idx]

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    features_2d = reducer.fit_transform(features)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))

    for label, color in zip(np.unique(labels), colors):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=[color], label=dataset.idx_to_class[label],
                    alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    plt.title('Latent Space (UMAP)', fontsize=16, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = 'outputs/quick_analysis.png'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example.py detect <image_path>     # Detect cells with YOLO")
        print("  python example.py similar <crop_path>     # Find similar cells")
        print("  python example.py analyze                 # Quick latent space analysis")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'detect':
        if len(sys.argv) < 3:
            print("Error: provide image path")
            sys.exit(1)
        detect_and_visualize(sys.argv[2])

    elif mode == 'similar':
        if len(sys.argv) < 3:
            print("Error: provide crop image path")
            sys.exit(1)
        find_similar_cells(sys.argv[2])

    elif mode == 'analyze':
        quick_analysis()

    else:
        print(f"Unknown mode: {mode}")
        print("Use: detect, similar, or analyze")
        sys.exit(1)
